# -*- coding: utf-8 -*-

import threading
from classify2 import Net
from car_faster import faster
import argparse
import caffe,os
from fast_rcnn.config import cfg
import numpy as np

CLASSES = ('__background__',
           'car')

NETS = {'vgg_cnn_m_1024':('VGG_CNN_M_1024','vgg_cnn_m_1024_faster_rcnn_iter_100000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_blur_iter_10000.caffemodel','vgg_cnn_m_1024_faster_rcnn_blur_train_iter_20000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_truck_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_dark_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all1_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_alldark_iter_50000.caffemodel'),
    'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_80000.caffemodel','vgg16_faster_rcnn_dark_iter_50000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def run_thread(net,car_det, im_names,clas):
    #net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print im_name
        print 'Demo for data/demo/{}'.format(im_name)
        if int(im_name.split('/')[1].split('.')[0])%10 == 0:
            car_det.split_wnd(net, im_name.strip('\n'),clas)
caffe.set_mode_gpu()
cfg.TEST.HAS_RPN = True  # Use RPN for proposals
cfg.TEST.SCALES = (410,)
clas = Net()
save_path = os.path.join('/home/sedlight/workspace/zjf/py-faster-rcnn/nvidia/data/out_img/test/')
image_path = os.path.join('/home/sedlight/workspace/wei/data/CITYCHANLLENGE/all_imgs/bg/')
car_det = faster(save_path,image_path)
car_det1 = faster(save_path,image_path)
args = car_det.parse_args()
args1 = car_det1.parse_args()
#caffe.set_mode_gpu()
#caffe.set_device(0)
#cfg.GPU_ID = 0
#cfg.TEST.HAS_RPN = True  # Use RPN for proposals
#cfg.TEST.SCALES = (410,)
#clas = Net()
prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.DATA_DIR,
                              NETS[args.demo_net][2])

prototxt1 = os.path.join(cfg.MODELS_DIR, NETS[args1.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel1 = os.path.join(cfg.DATA_DIR,
                              NETS[args1.demo_net][2])

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

#if args.cpu_mode:
#    caffe.set_mode_cpu()
#else:
#    caffe.set_mode_gpu()
#    caffe.set_device(0)
#    cfg.GPU_ID = 0
net1 = caffe.Net(prototxt, caffemodel, caffe.TEST)
#caffe.set_device(0)
#cfg.GPU_ID = 0
net2 = caffe.Net(prototxt1, caffemodel1, caffe.TEST)
print '\n\nLoaded network {:s}'.format(caffemodel)

# Warmup on a dummy image
#im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
#for i in xrange(2):
#    _, _= im_detect(net1, im)
#    _, _ = im_detect(net2, im)
print('test...')
threads = []
with open('track2-bg.txt','r') as f:
    lines = f.readlines()
thread1 = threading.Thread(target = run_thread, args = (net1,car_det,lines[:len(lines)/10],clas))
thread2 = threading.Thread(target = run_thread, args = (net2,car_det1,lines[len(lines)/10:2*len(lines)/10],clas))
threads.append(thread1)
threads.append(thread2)
for t in threads:
    t.start()
for t in threads:
    t.join()

