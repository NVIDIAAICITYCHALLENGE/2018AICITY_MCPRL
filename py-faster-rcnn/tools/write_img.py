# -*- coding: utf-8 -*-
import cv2 as cv

with open('pipei1.txt','r') as f:
    lines = f.readlines()
with open('pipei2.txt','r') as f1:
    lines1 = f1.readlines()
for i in range(len(lines)):
    if int(lines[i].split('/')[0]) == 49:
        im1 = cv.imread('/home/sedlight/workspace/wei/data/CITYCHANLLENGE/all_imgs/bg/' + lines[i].split(' ')[0])
        im2 = cv.imread('/home/sedlight/workspace/wei/data/CITYCHANLLENGE/all_imgs/bg/' + lines1[i].split(' ')[0])
        num = int((len(lines[i].split(' ')) - 2)/ 5)
        for j in range(1):
            bbox = lines[i].split(' ')[1 + 5*j:6 + 5*j]
            bbox1 = lines1[i].split(' ')[1 + 5*j:6 + 5*j]
            cv.rectangle(im1,(int(float(bbox[0])),int(float(bbox[1]))),(int(float(bbox[2])),int(float(bbox[3]))),(0,0,255),1)
            cv.rectangle(im2,(int(float(bbox1[0])),int(float(bbox1[1]))),(int(float(bbox1[2])),int(float(bbox1[3]))),(0,0,255),1)
        if 1:
            cv.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/test_sim/' + lines[i].split('/')[0] + '_' + lines[i].split('/')[1].split('.')[0] + '_1.jpg',im1)
            cv.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/test_sim/' + lines1[i].split('/')[0] + '_' + lines1[i].split('/')[1].split('.')[0] + '_2.jpg',im2)



