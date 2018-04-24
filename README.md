# Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling
This work participates in Nvidia AI City Challenge CVPR 2018 Workshop.

This repository contains our source code of Track-2 in the NVIDIA AI City Challenge at CVPR 2018 Workshop. 

## Introduction
The pipeline of our system is as follow: 

![](whole_system_new.png)

As shown in the picture, we get the background frames from the original frames. Then we detect vehicles in background images using Faster-RCNN. After that, to eliminate false detected bounding boxes, we utilize VGG as a powerful classifier. We determine anomalous vehicles
based on the bounding boxes produced by the former module. When meeting camera movement or vehicles waiting for red lights, we will compare the similarity between two candidates with the help of ResNet50 trained with triplet loss. For every detected abnormal vehicles, we go back to find the accurate timestamp when the abnormal happens. More details can be found in our paper.

## Dependencies
* Python2
* Caffe 
* pytorch
* cv2
* PIL
* Numpy

## Code structure

Because it takes a lone time to run our system on the entire evaluation dataset, which contains about 1500 minuntes video in total, we split and run our system in three part. What's more, it makes debugging easier. First, we generate the background frames. Next, we detecte vechiles using faster-rcnn in multi-scale, which is follower by a VGG16 classifier. At last, we obtain all anomalies according to saved bounding boxes. If you want, you can also run the whole system end-to-end, but you have to combine the code together. When meeting some trouble, you should feel free to [contact](#Contact) me

1. Run `python ./extract_background.py`. Then, you will get all original frames and background frames saved in `./data/all_imgs/all` and `./data/all_imgs/bg` separately, which are both used in the following modules.
2. The `./classification` contains the code to train classification model. You can also download our model rather than train it yourself.
3. The `./similar` contains the code to train similarity comparison model. You can also download our model rather than train it yourself.
4. The `./data` is the directory to save all images and models used in our work. [This](https://drive.google.com/open?id=1K18W1Zoj3hQI7BiQLqs-g-Ay6CKiMDbS) is some of our data for classification and the source other data are illustruted [here](./classification). [Here](./data/models) you can download all models used by us.
5. The `./py-faster-rcnn` contains almost all the scripts of our system.

More details can be found in corresponding subfolders.

## Reference
JiaYi Wei, JianFei Zhao, YanYun Zhao, ZhiCheng Zhao, "Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling", in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recogn. Workshops (CVPRW), Jun. 2018.


## Contact
If you have any question about our work, you can contact [JiaYi Wei](https://jiayi-wei.github.io/#contact)
