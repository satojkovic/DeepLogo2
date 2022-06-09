import io
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from six import BytesIO
from tqdm import tqdm

from config import CROPPED_ANNOT_FILE, DATA_DIR, IMAGES_DIR


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def convert_csv_into_numpy_array_rects_idxes(csv, im_width, im_height):
    # [xmin_0,ymin_0,xmax_0,ymax_0,cls_idx_0,...,xmin_N,ymin_N,xmax_N,ymax_N,cls_idx_N]
    elems = list(map(int, csv))
    rects, cls_idxes = [], []
    for xmin_pos_idx in range(0, len(elems), 5):
        xmin, ymin, xmax, ymax, cls_idx = elems[xmin_pos_idx : xmin_pos_idx + 5]
        xmin /= im_width
        ymin /= im_height
        xmax /= im_width
        ymax /= im_height
        rects.append([ymin, xmin, ymax, xmax])
        cls_idxes.append(cls_idx)
    return np.array(rects, dtype=np.float32), np.array(cls_idxes, dtype=np.int32)


def parse_csvs(annot_csv):
    csvs = []
    with open(annot_csv, 'r') as f:
        for line in f:
            line = line.rstrip().split(',')
            csvs.append(line)
    return csvs


def is_include(csv, target_idxes):
    target_idxes = set(target_idxes)
    elems = list(map(int, csv))
    for xmin_pos_idx in range(0, len(elems), 5):
        _, _, _, _, cls_idx = elems[xmin_pos_idx : xmin_pos_idx + 5]
        if cls_idx in target_idxes:
            return True
    return False


if __name__ == '__main__':
    train_annot_csv = CROPPED_ANNOT_FILE
    train_img_dir = IMAGES_DIR

    train_images_np = []
    gt_boxes = []
    gt_class_ids = []
    train_image_names = []
    csvs = parse_csvs(train_annot_csv)

    for csv in tqdm(csvs):
        img_fname = csv[0]
        train_image_names.append(img_fname)
        with tf.io.gfile.GFile(os.path.join(train_img_dir, img_fname), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        train_images_np.append(load_image_into_numpy_array(os.path.join(train_img_dir, img_fname)))
        rects, cls_idxes = convert_csv_into_numpy_array_rects_idxes(csv[1:], width, height)
        gt_boxes.append(rects)
        gt_class_ids.append(cls_idxes)

    save_dir = DATA_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'train_images_np.npy'), train_images_np)
    np.save(os.path.join(save_dir, 'gt_boxes.npy'), gt_boxes)
    np.save(os.path.join(save_dir, 'gt_class_ids.npy'), gt_class_ids)
