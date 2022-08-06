import argparse
import json
from collections import OrderedDict, defaultdict

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


def get_annots(cfg):
    annots = defaultdict(list)
    with open(cfg.CROPPED_ANNOT_FILE, 'r') as f:
        for line in f:
            elems = line.rstrip().split(',')
            jpg_file, groundtruth = elems[0], list(map(int, elems[1:]))
            annots[jpg_file] = groundtruth
    return annots


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', dest='cfg_file',
                        default=None, type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    annots = get_annots(cfg)

    query_list = ['info', 'licenses', 'images',
                  'annotations', 'categories', 'segment_info']
    js = OrderedDict()
    for i, query in enumerate(query_list):
        tmp = ''
        if query == 'info':
            tmp = info()
        if query == 'licenses':
            tmp = licences()

        js[query] = tmp

    with open('flickr_logos_27.json', 'w') as f:
        json.dump(js, f, indent=2)
