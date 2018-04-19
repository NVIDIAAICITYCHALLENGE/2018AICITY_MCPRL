# Train Model for Classifying Vehicle and Non-vehicle

## How it works

1. Prepare data. Put all your images (including vehicle and non_vehicle) in the `data/vehicle`. 
The data used to train our VGGNet comes from the following sources: 

    i)ImageNet (categories: car, truck, sign, road, snow and traffic light)
    ii) UIUC Car Detection 
    iii) GTI dataset 
    iv) Cars Dataset 
    v) Images of vehicles and non-vehicles randomly captured from the Extra-Video. Which is can be find [here](https://drive.google.com/drive/folders/1K18W1Zoj3hQI7BiQLqs-g-Ay6CKiMDbS?usp=sharing).
    For vehicle images, we crop the bounding boxes if provided and randomly crop 80\% of the original size during training. For non-vehicle images, we make a crop of random size (10%, 30%, 50%, 100%) of the original size during training. All training images are resized to 64x64 and are rotated a certain degree randomly chosen from (-10, 10).
