# Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling
This work participates in Nvidia AI City Challenge CVPR 2018 Workshop.

This repository contains our source code of Track-2 in the NVIDIA AI City Challenge at CVPR 2018 Workshop. 

## Introduction
The pipeline of our system is as follow: ![](whole_system_new.png)
As shown in the picture, we get the background frames from the original frames. Then we detect vehicles in background images using Faster-RCNN. After that, to eliminate false detected bounding boxes, we utilize VGG as a powerful classifier. We determine anomalous vehicles
based on the bounding boxes produced by the former module. When meeting camera movement or vehicles waiting for red lights, we will compare the similarity between two candidates with the help of ResNet50 trained with triplet loss. For every detected abnormal vehicles, we go back to find the accurate timestamp when the abnormal happens.

## Dependencies
* Faster-RCNN. 
* pytorch
* cv2
* PIL

## Code structure


## Reference


## Contact
If you have any question about our work, you can contact [JiaYi Wei](https://jiayi-wei.github.io/#contact)
