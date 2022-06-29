from data_loader import load_npy_data
from config import get_cfg_defaults
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

import matplotlib
matplotlib.use('Agg')


def data_preproc(train_images_np, gt_boxes, gt_class_ids, category_index):
    # Convert class labels to one-hot; convert everything to tensors.
    # The `label_id_offset` here shifts all classes by a certain number of indices;
    # we do this here so that the model receives one-hot labels where non-background
    # classes start counting at the zeroth index.  This is ordinarily just handled
    # automatically in our training binaries, but we need to reproduce it here.
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []

    # for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):
    for (train_image_np, gt_box_np, gt_class_id_np) in zip(train_images_np, gt_boxes, gt_class_ids):

        # convert training image to tensor, add batch dimension, add to list
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            train_image_np, dtype=tf.float32), axis=0))

        # convert numpy array to tensor, add to list
        gt_box_tensors.append(tf.convert_to_tensor(
            gt_box_np, dtype=tf.float32))

        # zero indexed ground truth
        # zero_indexed_groundtruth_classes = tf.convert_to_tensor(
        #    np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)

        # ground truth indexes (multi classes)
        # e.g. three logos(class id=[1, 0, 1], num_classes=2) in single image, np.ones=[1, 1, 1] * [1, 0, 1] => [1, 0, 1]
        #      then one hot encoding => [[0, 1], [1, 0], [0, 1]]
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.ones(shape=[gt_box_np.shape[0]],
                    dtype=np.int32) * gt_class_id_np
        )

        # do one hot encoding
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, len(category_index)))
    print('Done prepping data.')


def gen_category_index(cfg):
    category_index = {}
    for i, class_name in enumerate(cfg.CLASS_NAMES):
        category_index[i + 1] = {'id': i + 1, 'name': class_name}
    return category_index


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
      image_np: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      figsize: size for the figure.
      image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='Path to config file', default=None, type=str)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Test load_npy_data
    train_images_np, gt_boxes, gt_class_ids = load_npy_data(cfg)

    # Category index
    category_index = gen_category_index(cfg)

    #
    # Visualize ground truth box
    #

    # give boxes a score of 100%
    dummy_scores = [np.array([1.0] * gt_box.shape[0],
                             dtype=np.float32) for gt_box in gt_boxes]

    # define the figure size
    plt.figure(figsize=(30, 15))

    # use the `plot_detections()` utility function to draw the ground truth boxes
    label_id_offset = 1
    for idx in range(10):
        plt.subplot(2, 5, idx+1)
        plot_detections(
            train_images_np[idx],
            gt_boxes[idx],
            np.ones(shape=[gt_boxes[idx].shape[0]], dtype=np.int32) *
            gt_class_ids[idx] + label_id_offset,
            dummy_scores[idx], category_index)

    plt.savefig('data_preprocessor.png')
