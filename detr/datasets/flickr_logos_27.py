from pathlib import Path

from .coco import CocoDetection, make_coco_transforms


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided root path {root} does not exist'
    train_json = 'flickr_logos_27_train.json'
    test_json = 'flickr_logos_27_test.json'
    PATHS = {
        "train": (root / 'flickr_logos_27_dataset_images', root / train_json),
        "val": (root / 'flickr_logos_27_dataset_images', root / test_json),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
