"""
Mask R-CNN
Train on the satellite dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 sat.py train --dataset=/path/to/sat/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 sat.py train --dataset=/path/to/sat/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 sat.py train --dataset=/path/to/sat/dataset --weights=imagenet

    # Apply color splash to an image
    python3 sat.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 sat.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import time
import numpy as np
import skimage.draw
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Label configs
MASK_COLOURS = [[183, 28, 28], [49, 27, 146], [230, 81, 0], [51, 105, 30]]
NUM_LABELS = 2 # Including background

############################################################
#  Configurations
############################################################
np.set_printoptions(threshold=sys.maxsize)

class SatelliteConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sat"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = NUM_LABELS

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class SatelliteDataset(utils.Dataset):

    def load_satellite(self, dataset_dir, subset):
        """Load a subset of the Satellite dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add clases
        label_dirs = [ldir for ldir in os.listdir(os.path.join(dataset_dir, "labels"))
                        if os.path.isdir(os.path.join(dataset_dir, "labels", ldir))]
        for i, ldir in enumerate(label_dirs):
            self.add_class("sat", i+1, ldir)


        if subset == 'train':
            label = 0
        elif subset == 'val':
            label = 1
        elif subset == 'test':
            label = 2
        else:
            raise ValueError(
                'subset = %s not recognized.' % subset)

        # Train or validation dataset?
        subset_file = os.path.join(dataset_dir, 'subsets', 'label_subsets.txt')



        # Add images
        with open(subset_file, 'r') as f:
            for line in f.readlines()[1:]:
                words = line.split()
                if int(words[1]) == label:
                    filename = words[0]
                    image_path = os.path.join(dataset_dir, "img", filename)
                    mask_paths = [os.path.join(dataset_dir, "labels", ldir, filename)
                                    for ldir in label_dirs]

                    self.add_image(
                        "sat",
                        image_id=filename,  # use file name as a unique image id
                        path=image_path,
                        mask_paths=mask_paths)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a sat dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "sat":
            return super(self.__class__, self).load_mask(image_id)

        # Load mask of shape
        # [height, width, instance_count]
        masks = []
        for mask_path in info['mask_paths']:
            mask_img = skimage.io.imread(mask_path, as_gray=True)
            # Create a mask of 0s or 1s, depending on the intensity of the image
            mask = np.zeros_like(mask_img, dtype=int)[..., np.newaxis]
            mask[mask_img > 0.5] = 1
            masks.append(mask)
        masks = np.concatenate(masks, axis=-1)

        # Return mask, and array of class IDs of each instance.
        return masks, np.arange(1, masks.shape[-1]+1, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sat":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = SatelliteDataset()
    dataset_train.load_satellite(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SatelliteDataset()
    dataset_val.load_satellite(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')


def test(model, dataset, truth, results_dir, eval_type="segm", limit=0):
    """ Test model on dataset
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Pick images from the dataset
    image_ids = dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding satellite image IDs.
    sat_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_results(dataset, sat_image_ids[i],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)
        masked_img = image
        for j, class_id in enumerate(r['class_ids']):
            mask = r['masks'][...,j]
            # Apply mask
            colour = MASK_COLOURS[class_id % len(MASK_COLOURS)]
            masked_img = apply_mask_to_image(masked_img, mask, colour)
        # Save output
        file_name = os.path.join(results_dir, sat_image_ids[i])
        skimage.io.imsave(file_name, masked_img)
        results_file = os.path.splitext(os.path.join(results_dir, sat_image_ids[i]))[0] + ".txt"
        with open(results_file, 'w') as f:
            print(image_results, file=f)

    # Timing
    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


def apply_mask_to_image(image, mask, colour):
    # Set the mask colour to red
    # Create an image from the mask
    mask_img = np.zeros_like(image)
    mask_img[mask > 0.5] = colour
    # Combine the image with the mask
    combined = 0.4*image + 0.6*mask_img
    # Recreate the image
    image_output = image.copy()
    image_output[mask[...,0] > 0.5] = combined[mask[...,0] > 0.]
    return image_output


def build_results(dataset, image_id, rois, class_ids, scores, masks):
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    # Loop through detections
    for i in range(rois.shape[0]):
        class_id = class_ids[i]
        score = scores[i]
        bbox = np.around(rois[i], 1)
        mask = masks[:, :, i]

        result = {
            "image_id": image_id,
            "category_id": dataset.get_source_class_id(class_id, "sat"),
            "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
            "score": score,
            "segmentation": mask
        }
        results.append(result)
    return results


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect solar panels.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/sat/dataset/",
                        help='Directory of the Satellite dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--results', required=False,
                        default=DEFAULT_RESULTS_DIR,
                        metavar="/path/to/results/",
                        help='Results directory (default=results/)')
    parser.add_argument('--limit', required=False,
                        default=0,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=0)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SatelliteConfig()
    else:
        class InferenceConfig(SatelliteConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        dataset_val = SatelliteDataset()
        data = dataset_val.load_satellite(args.dataset, "test")
        dataset_val.prepare()
        print("Running test on {} images.".format(args.limit))
        test(model, dataset_val, data, args.results, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
