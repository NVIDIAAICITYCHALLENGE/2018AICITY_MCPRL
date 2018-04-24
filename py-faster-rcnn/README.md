# Detecting and Decision Making

## Introduction
This directory is modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 

## How it works

1. Build caffe, follow the introdution of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#installation-sufficient-for-the-demo). By the way, you can find out how to use the py-faster-rcnn project at the same link.
2. Train the detection model(optional). You can either [download](https://drive.google.com/open?id=1mCMV3pNt6RZEH_ywyA39rWJCUs4oZReM) our model. We train the faster-rcnn model with [DETRAC](https://detrac-db.rit.albany.edu/) dataset and [Extra-Video](https://drive.google.com/open?id=1K18W1Zoj3hQI7BiQLqs-g-Ay6CKiMDbS) dataset, which is collected and annotate by ourselves.
3. Detect vehicles in background frames, using `python tools/car-faster.py --net vgg16`. If you train your own detection model with other pre-trained model, you should modify the command line as well.
4. Run `python tools/repair_txt.py` to do double check before making decision about abnormal vehicles
5. Run `python tools/accident_time.py` to obtain the timestamp of every anomaly.

## Code structure

 * `caffe-fater-rcnn` is cloned from py-faster-rcnn.
 * `data` is the folder to save our detection model.
 * `model` is where to save `prototxt` for training and testing.
 * `nvidia` is where to save "txt" for training. It may be useless for you.
 * `tools` is the main directory for our system, and almost all useful scripts are saved here. `car_faster.py` is for detecting vehicles. Also, `similar.py` and `classify.py` are for similar comparison and classifying separately and called in `car_faster.py`. Using `accident_time.py`, accurate timestamp of an anomaly can be obtained according to bounding boxes.
