import argparse
from collections import defaultdict
from config import get_cfg_defaults


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='Path to config file', default=None, type=str)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    classids = []
    with open(cfg.CROPPED_ANNOT_FILE, 'r') as f:
        for line in f:
            elems = line.rstrip().split(',')
            ids = list(map(int, elems[5::5]))
            classids.extend(ids)

    classid_counts = defaultdict(int)
    for classid in classids:
        classid_counts[classid] += 1

    for classid in sorted(classid_counts.keys()):
        print(f'class {classid}: {classid_counts[classid]}')
