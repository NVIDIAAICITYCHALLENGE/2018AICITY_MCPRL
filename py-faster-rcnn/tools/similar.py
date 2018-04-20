import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import models
from PIL import Image
import os
import copy
import numpy as np
import torch.nn.functional as F
import cv2

imgtransforms=transforms.Compose([
    transforms.CenterCrop(size=(160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ])

class extract_model(nn.Module):
    def __init__(self):
        super(extract_model,self).__init__()
        model=models.resnet50(pretrained=False)
        self.model=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,
        model.layer2,model.layer3,model.layer4,nn.AvgPool2d(kernel_size=(5,5)))
        self.fc=nn.Linear(2048,512)

    def forward(self,inputs):
        inputs=self.model(inputs)
        outputs=inputs.view(-1,2048)
        outputs=self.fc(outputs)

        outputs=F.normalize(outputs, p=2, dim=1)
        return outputs

class GetSimilar():
    def __init__(self):
        model=extract_model()
        model.load_state_dict(torch.load('./../../data/models/trip_res50_34999_batch_all.pkl',map_location={'cuda:1':'cuda:0'}))
        model.eval()
        model=model.cuda()
        self.net = model

    def similar(self, img1, img2):
        #convert img_cv to tensor
        img1_cv = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_cv = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_pil = Image.fromarray(img1_cv)
        img2_pil = Image.fromarray(img2_cv)
        img1_ = imgtransforms(img1_pil.resize((160, 160))).resize_(1,3,160,160).cuda()
        img2_ = imgtransforms(img2_pil.resize((160, 160))).resize_(1,3,160,160).cuda()
        #get the feature
        result1 = self.net(Variable(img1_)).data.cpu().numpy()
        result2 = self.net(Variable(img2_)).data.cpu().numpy()
        return np.linalg.norm(result1-result2)



if __name__=='__main__':
    '''test the class'''
    with open('pipei1.txt','r') as f1:
        lines = f1.readlines()
    with open('pipei2.txt','r') as f2:
        lines2 = f2.readlines()
    for i in range(len(lines)):
        if int(lines[i].split('/')[0]) != 58:
            continue
        box1 = lines[i].strip('\n').split(' ')
        box2 = lines2[i].strip('\n').split(' ')
        img1 = cv2.imread('./../../data/all_imgs/bg/' + box1[0])
        img2 = cv2.imread('./../../data/all_imgs/bg/' + box2[0])
        img1 = img1[int(float(box1[2])):int(float(box1[4])),int(float(box1[1])):int(float(box1[3]))]
        img2 = img2[int(float(box2[2])):int(float(box2[4])),int(float(box2[1])):int(float(box2[3]))]
        Compare = GetSimilar()
        c = Compare.similar(img1, img2)
        #if c < 0.6:
        print 'true'
        cv2.imwrite('./../' + str(i) +'_'+str(c) + '_1'  + '.jpg',img1)
        cv2.imwrite('./../' + str(i) + '_' + str(c) + '_2.jpg',img2)
