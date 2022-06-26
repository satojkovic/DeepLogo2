#!/usr/bin/env python
# -*- coding=utf-8 -*-

from object_detection.utils import dataset_util
import argparse
import os
import numpy as np
import io
from PIL import Image
from config import get_cfg_defaults

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_test', required=True,
                        help='Generate tfrecord for train or test')
    parser.add_argument('--csv_input', required=True,
                        help='Path to the csv input')
    parser.add_argument('--output_path', required=True,
                        help='Path to output tfrecord')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='Path to config file', default=None, type=str)
    return parser.parse_args()


def create_tf_example(csv, cfg):
    img_fname = csv[0]
    x1, y1, x2, y2 = list(map(int, csv[1:-1]))
    cls_idx = int(csv[-1])
    cls_text = cfg.CLASS_NAMES[cls_idx].encode('utf8')
    with tf.gfile.GFile(os.path.join(cfg.IMAGE_DIR, img_fname), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    xmin = [x1 / width]
    xmax = [x2 / width]
    ymin = [y1 / height]
    ymax = [y2 / height]
    cls_text = [cls_text]
    cls_idx = [cls_idx]

    filename = img_fname.encode('utf8')
    image_format = b'jpg'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(cls_text),
        'image/object/class/label': dataset_util.int64_list_feature(cls_idx),
    }))

    return tf_example


if __name__ == "__main__":
    args = parse_arguments()
    train_or_test = args.train_or_test.lower()
    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    writer = tf.python_io.TFRecordWriter(args.output_path)
    csvs = np.loadtxt(args.csv_input, dtype=str, delimiter=',')
    img_fnames = set()
    num_data = 0
    for csv in csvs:
        img_fname = csv[0]
        tf_example = create_tf_example(csv, cfg)
        if train_or_test == 'train' or (train_or_test == 'test' and not img_fname in img_fnames):
            writer.write(tf_example.SerializeToString())
            num_data += 1
        img_fnames.add(img_fname)

    writer.close()
    print('Generated ({} imgs): {}'.format(num_data, args.output_path))
