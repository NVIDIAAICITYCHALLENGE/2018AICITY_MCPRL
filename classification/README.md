# Train Model for Classifying Vehicle and Non-vehicle

## How it works

### 0.Dependencies
Our code has been test on Ubuntu 14.04. You should have [pytorch](http://pytorch.org/) and [tensorbordX] on server, and tensorboardX is optional, which you can ignore in `train.py`. By the way, we test our code on pytorch 0.3.1, so we are not not sure about lower version.

### 1. Prepare data. 
Put all your images (including vehicle and non_vehicle) in the `data/vehicle`. The data used to train our VGGNet comes from the following sources: 

    * ImageNet (categories: car, truck, sign, road, snow and traffic light)
    * UIUC Car Detection 
    * GTI dataset 
    * Cars Dataset 
    * Images of vehicles and non-vehicles randomly captured from the Extra-Video. Which is can be find [here](https://drive.google.com/open?id=1K18W1Zoj3hQI7BiQLqs-g-Ay6CKiMDbS).
    
For vehicle images, we crop the bounding boxes if provided and randomly crop 80\% of the original size during training. For non-vehicle images, we make a crop of random size (10%, 30%, 50%, 100%) of the original size during training. All training images are resized to 64x64 and are rotated a certain degree randomly chosen from (-10, 10).

### 2.Get train.txt and test.txt
