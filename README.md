# ChannelAug
ChannelAug: A New Approach of Augmentation Method To Improve Robustness and Uncertainty

## Introduction
We propose a new data augmentation technique by separating the RGB channels of the image to improve the image classification ability in the uncertain environment of the image classification model. Many data augmentation techniques have been used to improve the image classification ability of models. For example, Flipping, Cropping etc. However, these data augmentation techniques are effective in improving image classification, but are not good in uncertain conditions. In order to solve this problem, we propose a ChannelAug that technique to improves robustness and uncertainty estimates ability of image classifier. and we compare other proposed image augmentation methods to show that ChannelAug can improves robustness and uncertainty measures on image classification.

## Requirements

*   numpy>=1.15.0
*   Pillow>=6.1.0
*   torch==1.2.0
*   torchvision==0.2.2

## Usage

Wide ResNet: `python train.py`

ResNeXt: `python train.py -m resnext`

DenseNet: `python train.py -m densenet`

## Download CIFAR C for Experiments

    CIFAR-10-C: https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    CIFAR-100-C: https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
    Default Path => ./data/cifar/CIFAR-10-C or ./data/cifar/CIFAR-100-C
    *you can change the corruption images path use "--corruption_path" option.

## Results

### Wide ResNet 500 Epochs CIFAR-10C Results
<img align="center" src="figures/CIFAR-10Cmeans.PNG" width="750">

### Wide ResNet 500 Epochs CIFAR-10C ECE and UCE
<img align="center" src="figures/CIFAR-10CCalibration.PNG" width="750">

## Citation

todo

## Contect

titania7777@seoultech.ac.kr