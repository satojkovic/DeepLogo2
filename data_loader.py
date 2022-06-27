import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from six import BytesIO
from tqdm import tqdm

from config import get_cfg_defaults


def parse_csvs(annot_csv):
    csvs = []
    with open(annot_csv, 'r') as f:
        for line in f:
            line = line.rstrip().split(',')
            csvs.append(line)
    return csvs


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def convert_csv_into_numpy_array_rects_idxes(csv, im_width, im_height):
    # [xmin_0,ymin_0,xmax_0,ymax_0,cls_idx_0,...,xmin_N,ymin_N,xmax_N,ymax_N,cls_idx_N]
    elems = list(map(int, csv))
    rects, cls_idxes = [], []
    for xmin_pos_idx in range(0, len(elems), 5):
        xmin, ymin, xmax, ymax, cls_idx = elems[xmin_pos_idx: xmin_pos_idx + 5]
        xmin /= im_width
        ymin /= im_height
        xmax /= im_width
        ymax /= im_height
        rects.append([ymin, xmin, ymax, xmax])
        cls_idxes.append(cls_idx)
    return np.array(rects, dtype=np.float32), np.array(cls_idxes, dtype=np.int32)


def load_npy_data(cfg):
    train_img_npy = os.path.join(cfg.DATA_DIR, 'train_images_np.npy')
    gt_boxes_npy = os.path.join(cfg.DATA_DIR, 'gt_boxes.npy')
    gt_class_ids_npy = os.path.join(cfg.DATA_DIR, 'gt_class_ids.npy')

    train_images_np = [] if not os.path.exists(
        train_img_npy) else np.load(train_img_npy, allow_pickle=True)
    gt_boxes = [] if not os.path.exists(
        gt_boxes_npy) else np.load(gt_boxes_npy, allow_pickle=True)
    gt_class_ids = [] if not os.path.exists(
        gt_class_ids_npy) else np.load(gt_class_ids_npy, allow_pickle=True)
    train_image_names = []

    if len(train_images_np) == 0 or len(gt_boxes) == 0 or len(gt_class_ids) == 0:
        csvs = parse_csvs(cfg.CROPPED_ANNOT_FILE)
        for csv in tqdm(csvs):
            # if not is_include(csv[1:], target_idxes=[0]):
            #   continue
            img_fname = csv[0]
            train_image_names.append(img_fname)
            with tf.io.gfile.GFile(os.path.join(cfg.IMAGE_DIR, img_fname), 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size

            train_images_np.append(load_image_into_numpy_array(
                os.path.join(cfg.IMAGE_DIR, img_fname)))
            rects, cls_idxes = convert_csv_into_numpy_array_rects_idxes(
                csv[1:], width, height)
            gt_boxes.append(rects)
            gt_class_ids.append(cls_idxes)

    return train_images_np, gt_boxes, gt_class_ids


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
    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.figsize'] = [30, 15]

    for idx, train_image_np in enumerate(train_images_np[:10]):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(train_image_np)
    plt.show()
