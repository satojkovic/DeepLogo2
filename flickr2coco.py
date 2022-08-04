from collections import OrderedDict
import json


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


if __name__ == '__main__':
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
