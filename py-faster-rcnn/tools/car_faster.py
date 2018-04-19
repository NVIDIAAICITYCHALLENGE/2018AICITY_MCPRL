#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

from classify2 import Net
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys
import argparse
#from classify2 import Net
import cv2

#the classes
CLASSES = ('__background__',
           'car')

#the model name
NETS = {'vgg_cnn_m_1024':('VGG_CNN_M_1024','vgg_cnn_m_1024_faster_rcnn_iter_100000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_blur_iter_10000.caffemodel','vgg_cnn_m_1024_faster_rcnn_blur_train_iter_20000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_truck_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_dark_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all1_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_alldark_iter_50000.caffemodel'),
    'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_80000.caffemodel','vgg16_faster_rcnn_dark_iter_50000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
class faster():
    def __init__(self,save_path,image_path,number = 2,thresh = 0.9):
        self.save_path = save_path
        self.image_path = image_path
        self.number = number #the multi-scale's number
        self.thresh = thresh # the detect thresh
        self.clas = Net() # the classity model

    #get boundingbox
    def bbox_get1(self,dets,thresh=0.5):
        label =[]
        inds = np.where(dets[:,-1] >= thresh)[0]
        if len(inds) != 0:
            for i in inds:
                bbox = dets[i,:4]
                score = dets[i,-1]
                a = [bbox[0],bbox[1],bbox[2],bbox[3],score]
                label.append(a)
        return label

    #get multi-scale boundingbox, and resize
    def bbox_get(self,dets,number,size,sets,thresh=0.5):
        label = []
        inds = np.where(dets[:,-1] >= thresh)[0]
        if len(inds) != 0:
            for i in inds:
                bbox = dets[i,:4]
                score = dets[i,-1]
                a = [bbox[0] * sets[0]  + number//size*sets[2],bbox[1] * sets[1] + number%size*sets[3],bbox[2]*sets[0] + number//size*sets[2],bbox[3] * sets[1] + number % size*sets[3],score]
                label.append(a)
        return label

    #multi-scale detect
    def split_wnd(self,net,image_name,clas):
        im = cv2.imread(self.image_path + image_name)
        dets = self.demo(net,im)
        label = []
        bbox = self.bbox_get1(dets,self.thresh)
        if len(bbox) != 0:
            label.extend(bbox)
        wnd_lenth = [449,299,229,179]
        wnd_wedth = [254,169,139,101]
        size = (800,410)
        x_step,y_step = [350,250,190,155],[155,120,90,77]
        x,y,cont = 0,0,-1
        for i in range(self.number):
            x,cont = 0,-1
            while wnd_lenth[i] + x < 800 :
                while wnd_wedth[i] + y < 410:
                    wnd = im[y:y+wnd_wedth[i],x:x+wnd_lenth[i]]
                    wnd2 = cv2.resize(wnd,size,interpolation = cv2.INTER_LINEAR)
                    y += y_step[i]
                    cont += 1
                    dets = self.demo(net,wnd2)
                    bbox = self.bbox_get(dets,cont,i + 2,[(wnd_lenth[i] + 1)/800.0,(wnd_wedth[i] + 1)/410.0,x_step[i],y_step[i]],self.thresh)
                    if len(bbox) != 0:
                        label.extend(bbox)
                y = 0
                x += x_step[i]
        return self.vis_detections1(im,label,image_name,clas)


    def classification(self,im,bbox):
        rec = im[int(float(bbox[1])):int(float(bbox[3])),int(float(bbox[0])):int(float(bbox[2]))]
        return self.clas.classify(rec)

    #save the result
    def vis_detections1(self,im,label,image_name,clas):
        if len(label) == 0:
            return
        label = sorted(label,key = lambda l:l[-1],reverse = True)
        im1 = im.copy()
        exsit,result,b = [],[],0
        exsit.append(label[0])
        start,end = 0, len(label) - 1
        while start <= len(label) - 1:
            a = 0
            for i in exsit:
                if not(label[start][2] < i[0] or label[start][3] < i[1] or label[start][0] > i[2] or label[start][1] > i[3]):
                    [x0,x1,x2,x3] = sorted([label[start][0],label[start][2],i[0],i[2]])
                    [y0,y1,y2,y3] = sorted([label[start][1],label[start][3],i[1],i[3]])
                    s1 = (x2 - x1)*(y2 - y1)
                    s21,s22 = (label[start][2] - label[start][0])*(label[start][3] - label[start][1]), (i[2] - i[0]) * (i[3] - i[1])
                    s2 = s21 + s22 - s1
                    if s1/s2 > 0.3:
                        a = 1
                        break
            if a == 0:
                exsit.append(label[start])
            start += 1
        #write the result in txtfile,for the clas.txt we using the classify
        f1 = open('./txtfile/result_8_3*3_nclas.txt','a')
        with open('./txtfile/result_8_3*3_clas.txt','a') as f:
            f.write(image_name + ' ')
            f1.write(image_name + ' ')
            for rec in exsit:
                wnd = im[int(rec[1]):int(rec[3]),int(rec[0]):int(rec[2])]
                f1.write(str(rec[0]) + ' ' + str(rec[1]) + ' ' + str(rec[2]) + ' ' + str(rec[3]) + ' ' + str(rec[4]) + ' ')
                if clas.classify(wnd):
                    b = 1
                    result.append(rec)
                    f.write(str(rec[0]) + ' ' + str(rec[1]) + ' ' + str(rec[2]) + ' ' + str(rec[3]) + ' ' + str(rec[4]) + ' ')
                    #if you want to view the result in image, you can uncomment them
                    #cv2.rectangle(im,(int(rec[0]),int(rec[1])),(int(rec[2]),int(rec[3])),(0,0,255),1)
                    #cv2.rectangle(im1,(int(rec[0]),int(rec[1])),(int(rec[2]),int(rec[3])),(0,0,255),1)
                #else:
                    #cv2.rectangle(im1,(int(rec[0]),int(rec[1])),(int(rec[2]),int(rec[3])),(255,0,0),1)
            f.write('\n')
            f1.write('\n')
            #path = []
            #path.append(self.save_path + 'clas/')
            #path.append(self.save_path + 'nclas/')
            #map(lambda x:os.mkdir(x),[i for i in path if not os.path.exists(i)])
            #if b == 1:
            #    cv2.imwrite(path[0] + image_name.split('/')[0]+'_'+image_name.split('/')[1],im)
            #cv2.imwrite(path[1] + image_name.split('/')[0]+'_'+image_name.split('/')[1],im1)
        f1.close()
        return result

    #detect the image
    def demo(self, net, im):
        """Detect object classes in an image using pre-computed object proposals."""
        scores, boxes = im_detect(net, im)
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
        return dets

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Faster R-CNN demo')
        parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                            default=0, type=int)
        parser.add_argument('--cpu', dest='cpu_mode',
                            help='Use CPU mode (overrides --gpu)',
                            action='store_true')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                            choices=NETS.keys(), default='vgg_cnn_m_1024')

        args = parser.parse_args()

        return args

if __name__ == '__main__':
    #set the path, save_path will save the result iamges, image_path save the input data
    save_path = os.path.join('../out/')
    image_path = os.path.join('../data/track2-bg-imgs/')
    #get the model
    car_det = faster(save_path,image_path,2)
    #set gpu
    args = car_det.parse_args()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = (410,)
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR,
                              NETS[args.demo_net][2])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    #get the classification model
    clas = Net()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print ('\nLoaded network {:s}'.format(caffemodel))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    print('test...')

    timer = Timer()
    #test
    f = open('txtfile/test_bg_all.txt','r')
    lines = f.readlines()
    cont = 0
    for im_name in lines:
        cont += 1
        #we detect every five frame
        if int(im_name.split('/')[1].split('.')[0])%5 == 0:
            timer.tic()
            print(car_det.split_wnd(net, im_name.strip('\n'),clas))
            timer.toc()
            print(('Detection took {:.3f}s').format(timer.total_time))
    f.close()

