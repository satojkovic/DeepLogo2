import argparse
import json
from collections import OrderedDict, defaultdict
import cv2
import os

from config import get_cfg_defaults


def info():
    tmp = OrderedDict()
    tmp['description'] = 'Flickr logos 27 dataset'
    tmp['url'] = 'http://image.ntua.gr/iva/datasets/flickr_logos/'
    tmp['version'] = '1.0'
    tmp['year'] = '2022'
    tmp['contributor'] = 'satojkovic'
    tmp['data_created'] = '2022/08/03'
    return tmp


def licences():
    tmp = OrderedDict()
    tmp['id'] = 1
    tmp['url'] = 'Unknown'
    tmp['name'] = 'Unknown'
    return tmp


def images(image_idxes, image_sizes):
    tmps = []
    for k in image_idxes.keys():
        tmp = OrderedDict()
        tmp['licenses'] = 1
        tmp['id'] = image_idxes[k]
        tmp['file_name'] = k
        tmp['height'] = image_sizes[k][0]
        tmp['width'] = image_sizes[k][1]
        tmp['date_captured'] = ''
        tmp['coco_url'] = ''
        tmp['flickr_url'] = ''
        tmps.append(tmp)
    return tmps


def get_annots(cfg):
    annots = defaultdict(list)
    with open(cfg.CROPPED_ANNOT_FILE, 'r') as f:
        for line in f:
            elems = line.rstrip().split(',')
            jpg_file, groundtruth = elems[0], list(map(int, elems[1:]))
            annots[jpg_file] = groundtruth
    return annots


def get_image_idxes(annots):
    idxes = defaultdict(int)
    for i, k in enumerate(annots.keys()):
        idxes[k] = i
    return idxes


def get_image_sizes(annots, cfg):
    image_sizes = defaultdict(set)
    for k in annots.keys():
        img = cv2.imread(os.path.join(cfg.IMAGE_DIR, k))
        h, w, _ = img.shape
        image_sizes[k] = (h, w)
    return image_sizes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', dest='cfg_file',
                        default=None, type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    annots = get_annots(cfg)
    image_idxes = get_image_idxes(annots)
    image_sizes = get_image_sizes(annots, cfg)

    query_list = ['info', 'licenses', 'images',
                  'annotations', 'categories', 'segment_info']
    js = OrderedDict()
    for i, query in enumerate(query_list):
        tmp = ''
        if query == 'info':
            tmp = info()
        if query == 'licenses':
            tmp = licences()
        if query == 'images':
            tmp = images(image_idxes, image_sizes)

        js[query] = tmp

    with open('flickr_logos_27.json', 'w') as f:
        json.dump(js, f, indent=2)
