#!/usr/bin/env python
# -*- coding=utf-8 -*-

from config import get_cfg_defaults
import warnings
import numpy as np
import argparse
from collections import defaultdict


def save_annots(save_file_path, preprocessed_annots, keys):
    with open(save_file_path, 'w') as f:
        for key in keys:
            # img_name,x1_0,y1_0,x2_0,y2_0,cls_idx_0,...,x1_n,y1_n,x2_n,y2_n,cls_idx_n
            coords_and_idxes = [','.join(annot)
                                for annot in preprocessed_annots[key]]
            coords_and_idxes = ','.join(coords_and_idxes)
            f.writelines(','.join([key, coords_and_idxes]))
            f.writelines("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=6,
                        help='Subset number(defalut: 6)')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='Path to config file', default=None, type=str)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

        # Load annot file
    annots = []
    with open(cfg.ANNOT_FILE, 'r') as f:
        for line in f:
            elems = line.rstrip().split()
            annots.append(elems)
    print('Num. of annots:', len(annots))

    # Preprocess annot data and save preprocessed annots
    #  - Skip with size 0
    #  - Skip all but a subset of targets
    preprocessed_annots = defaultdict(list)
    for annot in annots:
        img_name, cls_name = annot[0], annot[1]
        cls_idx = cfg.CLASS_NAMES.index(cls_name)
        subset = int(annot[2])
        if subset != args.subset:
            continue
        x1, y1, x2, y2 = list(map(int, annot[3:]))
        w, h = (x2 - x1), (y2 - y1)
        if w == 0 or h == 0:
            print('Skip with size 0:', img_name)
            continue
        coords_and_idxes = [str(x1), str(y1), str(x2), str(y2), str(cls_idx)]
        preprocessed_annots[img_name].append(coords_and_idxes)
    print('Num. of preprocessed annots: {}(images)'.format(
        len(preprocessed_annots)))

    preprocessed_annots_keys = list(preprocessed_annots.keys())
    np.random.shuffle(preprocessed_annots_keys)
    num_train = int(len(preprocessed_annots) * 0.8)
    save_annots(cfg.CROPPED_ANNOT_FILE, preprocessed_annots,
                preprocessed_annots_keys[:num_train])
    save_annots(cfg.CROPPED_ANNOT_FILE_TEST, preprocessed_annots,
                preprocessed_annots_keys[num_train:])
    print('Num. of annotations: {}(train) {}(test)'.format(
        num_train, len(preprocessed_annots_keys) - num_train))
    print('Created: {}'.format(cfg.CROPPED_ANNOT_FILE))
    print('Created: {}'.format(cfg.CROPPED_ANNOT_FILE_TEST))


if __name__ == "__main__":
    with warnings.catch_warnings():
        # Supress low contrast warnings
        warnings.simplefilter("ignore")

        # Crop logo images
        main()
