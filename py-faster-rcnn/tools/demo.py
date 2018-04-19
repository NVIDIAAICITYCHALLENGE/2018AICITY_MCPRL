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

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from classify2 import Net

CLASSES = ('__background__',
           'car')

NETS = {'vgg_cnn_m_1024':('VGG_CNN_M_1024','vgg_cnn_m_1024_faster_rcnn_iter_100000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_blur_iter_10000.caffemodel','vgg_cnn_m_1024_faster_rcnn_blur_train_iter_20000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_truck_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_dark_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all1_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_alldark_iter_50000.caffemodel'),
    'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_80000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def bbox_get1(dets,thresh=0.5):
    label =[]
    inds = np.where(dets[:,-1] >= thresh)[0]
    if len(inds) != 0:
        for i in inds:
            bbox = dets[i,:4]
            score = dets[i,-1]
            a = [bbox[0],bbox[1],bbox[2],bbox[3],score]
            label.extend(a)
    return label

def bbox_get(dets,number,size,sets,thresh=0.5):
    label = []
    inds = np.where(dets[:,-1] >= thresh)[0]
    if len(inds) != 0:
        for i in inds:
            bbox = dets[i,:4]
            print(bbox,number,size,sets)
            score = dets[i,-1]
            print(bbox[0] * sets[0],number//size,number//size*sets[2])
            a = [bbox[0] * sets[0]  + number//size*sets[2],bbox[1] * sets[1] + number%size*sets[3],bbox[2]*sets[0] + number//size*sets[2],bbox[3] * sets[1] + number % size*sets[3],score]
            label.extend(a)
    return label

def vis_detections(im,image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    f1 = open('out_city.txt','a')
    a = image_name + ' ' + str(len(inds)) + ' '
    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        for j in range(5):
            a += str(dets[i,j]) + ' '
        print(bbox)

        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),1)
    #cv2.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/test/'+image_name[:9]+'_'+image_name[10:],im)

    cv2.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/city_split_img/' + image_name[:2] +
                '_' + image_name[3:],im)
    print(image_name[:2])
    f1.write(a + '\n')
    f1.close()
    #    ax.add_patch(
    #        plt.Rectangle((bbox[0], bbox[1]),
    #                      bbox[2] - bbox[0],
    #                      bbox[3] - bbox[1], fill=False,
    #                      edgecolor='red', linewidth=3.5)
    #        )
    #    ax.text(bbox[0], bbox[1] - 2,
    #            '{:s} {:.3f}'.format(class_name, score),
    #            bbox=dict(facecolor='blue', alpha=0.5),
    #            fontsize=14, color='white')

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()

def split_wnd(net,image_name,clas):
    im = cv2.imread('/home/sedlight/workspace/wei/data/CITYCHANLLENGE/track2-bg-imgs/'+image_name)
    #im = cv2.imread('/home/sedlight/workspace/zjf/py-faster-rcnn/'+image_name)
    dets = demo(net,im)
    label = []
    thresh = 0.9
    bbox = bbox_get1(dets,thresh)
    if len(bbox) != 0:
        label.append(bbox)
    wnd_lenth = [449,299,229,179]
    wnd_wedth = [254,169,139,101]
    size = (800,410)
    x_step,y_step = [350,250,190,155],[155,120,90,77]
    x,y,cont = 0,0,-1
    #split_label = [[[0,0,449,254],[350,0,799,254],
    #                [0,155,449,409],[350,155,799,409],[float(9.0/16),float(255.0/410),350,155]],
    #               [[0,0,299,169],[250,0,549,169],[500,0,799,169],
    #                [0,120,299,289],[250,120,549,289],[500,120,799,280],
    #                [0,240,299,409],[250,240,549,409],[500,240,799,409],[float(3.0/8),float(17.0/41),250,120]],
    #               [[0,0,229,139],[190,0,419,139],[380,0,609,139],[570,0,799,139],
    #                [0,90,229,229],[190,90,419,229],[380,90,609,229],[570,90,799,229],
    #                [0,180,229,319],[190,180,419,319],[380,180,609,319],[570,180,799,319],
    #                [0,270,229,409],[190,270,419,409],[380,270,609,409],[570,270,799,409],[float(23.0/80),float(14.0/41),190,90]]]
    for i in range(3):
        x,cont = 0,-1
        while wnd_lenth[i] + x < 800:
            while wnd_wedth[i] + y < 410:
    #for i in range(len(split_label)):
        #for j in range(len(split_label[i]) - 1):
                wnd = im[y:y+wnd_wedth[i],x:x+wnd_lenth[i]]
            #wnd = im[split_label[i][j][1]:split_label[i][j][3],split_label[i][j][0]:split_label[i][j][2]]
                wnd2 = cv2.resize(wnd,size,interpolation = cv2.INTER_LINEAR)
                y += y_step[i]
                cont += 1
                dets = demo(net,wnd2)
                bbox = bbox_get(dets,cont,i + 2,[(wnd_lenth[i] + 1)/800.0,(wnd_wedth[i] + 1)/410.0,x_step[i],y_step[i]],thresh)
                if len(bbox) != 0:
                    #print(bbox)
                    label.append(bbox)
            y = 0
            x += x_step[i]
    #print(x, y)
    vis_detections1(im,label,image_name,clas)

def vis_detections1(im,label,image_name,clas):
    if len(label) == 0:
        return
    b = 0
    exsit = []
    #print(label)
    #wnd = im[int(label[0][1]):int(label[0][3]),int(label[0][0]):int(label[0][2])]
    #if clas.classify(wnd):
    #    b = 1
    #    cv2.rectangle(im,(int(label[0][0]),int(label[0][1])),(int(label[0][2]),int(label[0][3])),(0,255,0),1)
    #else:
    #    cv2.rectangle(im,(int(label[0][0]),int(label[0][1])),(int(label[0][2]),int(label[0][3])),(255,0,0),1)
    exsit.append(label[0])
    start,end = 0, len(label) - 1
    #for box in label:
    while start < end:
        a = 0
        #while no_over:
            #no_over = 0
            #for j,i in enumerate(exsit):
        for j,i in enumerate(label[start + 1:]):
            if not(label[start][2] < i[0] or label[start][3] < i[1] or label[start][0] > i[2] or label[start][1] > i[3]):
                [x0,x1,x2,x3] = sorted([label[start][0],label[start][2],i[0],i[2]])
                [y0,y1,y2,y3] = sorted([label[start][1],label[start][3],i[1],i[3]])
                s1 = (x2 - x1)*(y2 - y1)
                s21,s22 = (label[start][2] - label[start][0])*(label[start][3] - label[start][1]), (i[2] - i[0]) * (i[3] - i[1])
                s2 = s21 + s22 - s1
                #print(s1,s2)
                if s1/s2 > 0.3:
                    label[start + j + 1] = [x0,y0,x3,y3,max(label[start][-1],i[-1])]
                    start += 1
                    a = 1
                    break

                    #no_over = 1
                    #a = 1
                    #if  s21 > s22 :
                    #if box[-1] > i[-1]:
                        #exsit[j] = (int(box[0]),int(box[1]),int(box[2]),int(box[3]),box[-1])
                        #exsit[j] = [x0,y0,x3,y3,max(box[-1],i[-1])]
                        #[box[0],box[1],box[2],box[3]] = exsit[j][:-2]
                        #break
        if a == 0:
            label[start],label[end] = label[end],label[start]
            end -= 1
            #wnd = im[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
            #if clas.classify(wnd):
                #b = 1
                #cv2.rectangle(im,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
            #else:
                #cv2.rectangle(im,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),1)
            #exsit.append((int(box[0]),int(box[1]),int(box[2]),int(box[3]),box[-1]))
    #print(exsit)
    exsit = label[start:]
    for rec in exsit:
        #print(rec)
        wnd = im[max(int(rec[1]),0):min(int(rec[3]),410),max(int(rec[0]),0):min(int(rec[2]),800)]
        print(rec)
        if clas.classify(wnd):
            b = 1
            cv2.rectangle(im,(int(rec[0]),int(rec[1])),(int(rec[2]),int(rec[3])),(0,255,0),1)
        else:
            cv2.rectangle(im,(int(rec[0]),int(rec[1])),(int(rec[2]),int(rec[3])),(255,0,0),1)
    if b == 1 and int(image_name.split('/')[0]) in [2,14,33,49,51,58,63,66,72,73,74,79,83,91,93,95,97]:
        cv2.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/vgg16_0.9_3/true/' + image_name.split('/')[0]+'_'+image_name.split('/')[1],im)
    elif int(image_name.split('/')[0]) in [2,14,33,49,51,58,63,66,72,73,74,79,83,91,93,95,97]:
        cv2.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/vgg16_0.9_3/truefalse/' + image_name.split('/')[0] + '_' + image_name.split('/')[1],im)
    elif b == 1:
        cv2.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/vgg16_0.9_3/falsetrue/' + image_name.split('/')[0] + ' ' + image_name.split('/')[1],im)
    #print('aaa')
    #cv2.imwrite('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/' + image_name.split('/')[1],im)

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join('/home/sedlight/workspace/wei/data/CITYCHANLLENGE/track2-bg-imgs/',image_name)
    #im = cv2.imread('/home/sedlight/workspace/wei/data/CITYCHANLLENGE/track2-bg-imgs/' + image_name)
    #wnd_lenth = 499
    #wnd_wedth = 254
    #size = (800,410)
    #x_step,y_step = 300,155
    #x,y,cont = 0,0,0
    #while wnd_lenth + x < 800:
    #    while y + wnd_wedth < 410:
    #        wnd = im[y:y+wnd_wedth,x:x+wnd_lenth]
    #        wnd = cv2.resize(wnd,size,interpolation = cv2.INTER_LINEAR)
    #        y += y_step
    #        cont += 1
     #Detect all object classes and regress object bounds
            #timer = Timer()
            #timer.tic()
    scores, boxes = im_detect(net, im)
            #timer.toc()
            #print ('Detection took {:.3f}s for '
            #    '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class

    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im,image_name[:-4]  + '.jpg', cls, dets, thresh=CONF_THRESH)
        #y = 0
        #x += x_step
    return dets

def parse_args():
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
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = (410,)
    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR,
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    print('test...')
    clas = Net()

    f = open('track2-bg.txt','r')
    lines = f.readlines()
    cont = 0
    im_names = ['nvidia/frame1800.jpg','nvidia/frame690.jpg','nvidia/frame75.jpg']
    for im_name in lines:
        cont += 1
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print im_name
        print 'Demo for data/demo/{}'.format(im_name)
        #name_split = im_name.strip('\n').split('/')
        #if int(im_name.split('/')[0]) in [3,4,2,51,63,73,33,11] and int(im_name.split('/')[1].split('.')[0])%5 == 0:
        #if int(im_name.split('/')[0]) in [14,33,51,63,66,73,83,91,95] and
        if int(im_name.split('/')[1].split('.')[0])%10 == 0:
        #if cont > 46742:
        #if os.path.exists('/home/sedlight/workspace/zjf/city_split_img2/' + im_name.strip('\n').split('/')[0] +
        #                  '_' +im_name.strip('\n').split('/')[1]):
            split_wnd(net, im_name.strip('\n'),clas)
    f.close()
    #plt.show()
