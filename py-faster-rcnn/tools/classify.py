# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torchvision import transforms
import models
import numpy as np
import os
from PIL import Image
import cv2
os.environ['CUDA_VISIBLE_DEVICES']='1'

root = '/home/sedlight/workspace/wei/AICity/vehicle_classification/logs'
experiment_name = 'vgg16'
model_name = 'model_20000.tar'
img_path = './1.jpg'
vehicle = [(631, 77), (655, 97)]
non_vhc = [(400, 200), (430, 230)]

class VehicleNet():
    def __init__(self, ex_name=experiment_name, m_name=model_name):
        net = models.Net(name=experiment_name)
        net.load_state_dict(torch.load(os.path.join(root, experiment_name, model_name)))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            net.cuda()
        self.net = net
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.cuda = use_cuda
    def classify(self, img_cv2):
        '''get IOU of image with Opencv
        and then load the IOU with PIL
        for classification'''
        #resize
        img_cv2 = cv2.resize(img_cv2, (64,64))
        #cv2 format to PIL format
        cv2_im = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        #To 4D Variable
        img = self.transforms(pil_im).unsqueeze(0)
        if self.cuda:
            img = img.cuda()
        img = Variable(img)
        #prediction
        pred = self.net.eval()(img)
        _, result = torch.max(pred.data, 1)
        result = result.cpu().numpy()
        return True if result==0 else False

if __name__=='__main__':
    net = Net()
    img = cv2.imread(img_path)
    img_v = img[vehicle[0][1]:vehicle[1][1], vehicle[0][0]:vehicle[1][0]]
    img_n = img[non_vhc[0][1]:non_vhc[1][1], non_vhc[0][0]:non_vhc[1][0]]
    #cv2.imwrite('a.jpg', img_v)
    #cv2.imwrite('b.jpg', img_n)
    print net.classify(img_v)
    print net.classify(img_n)
