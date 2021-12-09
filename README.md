# DeepLogo2

A brand logo detection system using Object Detection API with Tensorflow 2.
(DeepLogo with Tensorflow 1 is [here](https://github.com/satojkovic/DeepLogo))

# Description

DeepLogo supports Tensorflow 2 and its name is now DeepLogo2!

# Dataset

DeepLogo2 use the [flickr logos 27 dataset](http://image.ntua.gr/iva/datasets/flickr_logos/).
The flickr logos 27 dataset contains 27 classes of brand logo images downloaded from Flickr. The brands included in the dataset are: Adidas, Apple, BMW, Citroen, Coca Cola, DHL, Fedex, Ferrari, Ford, Google, Heineken, HP, McDonalds, Mini, Nbc, Nike, Pepsi, Porsche, Puma, Red Bull, Sprite, Starbucks, Intel, Texaco, Unisef, Vodafone and Yahoo.

```bash
$ wget http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz
$ tar zxvf flickr_logos_27_dataset.tar.gz
$ cd flickr_logos_27_dataset
$ tar zxvf flickr_logos_27_dataset_images.tar.gz
```
