import numpy as np
import os
import tensorflow as tf
from PIL import Image
from six import BytesIO
from tqdm import tqdm


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


def get_images(image_dir, annot):
    images_np = []
    csvs = np.loadtxt(annot, dtype=str, delimiter=',')
    for csv in tqdm(csvs):
        img_fname = csv[0]
        image_path = os.path.join(image_dir, img_fname)
        images_np.append(load_image_into_numpy_array(image_path))


if __name__ == '__main__':
    # Load images and visualize
    DATA_ROOT_DIR = 'flickr_logos_27_dataset'
    IMAGE_DIR = os.path.join(DATA_ROOT_DIR, 'flickr_logos_27_dataset_images')
    TRAIN_ANNOT_FILE = os.path.join(
        DATA_ROOT_DIR, 'flickr_logos_27_dataset_training_set_annotation_cropped.txt')
    TEST_ANNOT_FILE = os.path.join(
        DATA_ROOT_DIR, 'flickr_logos_27_dataset_test_set_annotation_cropped.txt')

    train_images_np = get_images(IMAGE_DIR, TRAIN_ANNOT_FILE)
    test_images_np = get_images(IMAGE_DIR, TEST_ANNOT_FILE)

    if not os.path.exists('data'):
        os.makedirs('data')

    np.save(os.path.join('data', 'train_images.npy'), train_images_np)
    np.save(os.path.join('data', 'test_images.npy'), test_images_np)
