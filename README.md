# Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling
This work participates in Nvidia AI City Challenge CVPR 2018 Workshop.

This repository contains our source code of Track-2 in the NVIDIA AI City Challenge at CVPR 2018 Workshop. 

## Introduction
The pipeline of our system is as follow: 

![](whole_system_new.png)

As shown in the picture, we get the background frames from the original frames. Then we detect vehicles in background images using Faster-RCNN. After that, to eliminate false detected bounding boxes, we utilize VGG as a powerful classifier. We determine anomalous vehicles
based on the bounding boxes produced by the former module. When meeting camera movement or vehicles waiting for red lights, we will compare the similarity between two candidates with the help of ResNet50 trained with triplet loss. For every detected abnormal vehicles, we go back to find the accurate timestamp when the abnormal happens. More details can be found in our paper.

## Dependencies
* Caffe 
* pytorch
* cv2
* PIL

## Code structure

1. The `./classification` contains the code to train classification model.
2. The `./similar` contains the code to train similarity comparison model.
3. The `./data` is the directory to save all images and models used in our work. [This](https://drive.google.com/open?id=1K18W1Zoj3hQI7BiQLqs-g-Ay6CKiMDbS) is some of our data for classification and the source other data are illustruted [here](./classification). [Here](./data/models) you can download all models used by us.
4. The `./py-faster-rcnn` contains all the scripts for detecting, classifiying, decision making, etc.

More details can be found in corresponding subfolders.

## Reference
JiaYi Wei, JianFei Zhao, YanYun Zhao, ZhiCheng Zhao, "Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling", in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn. Workshops (CVPRW), Jun. 2018.


## Contact
If you have any question about our work, you can contact [JiaYi Wei](https://jiayi-wei.github.io/#contact)
