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

class alex_trip(nn.Module):
    def __init__(self):
        super(alex_trip,self).__init__()
        model=models.alexnet(pretrained=False)
        self.model1=model.features
        # self.fc1=nn.Linear(1024,4096)
        self.fc1=nn.Linear(6400,4096)
        self.fc2=model.classifier[2]
        self.dropout=nn.Dropout(0)
        self.relu=nn.ReLU()
        self.fc=nn.Linear(4096,512)
    def forward(self,inputs):
        #print 'inputs',inputs.size()
        inputs=self.model1(inputs)
        
        #print inputs.size()
        # inputs=inputs.view(-1,1024)
        inputs=inputs.view(-1,6400)
        inputs=self.dropout(self.relu(self.fc1(inputs)))
        inputs=self.dropout(self.relu(self.fc2(inputs)))
        outputs=self.fc(inputs)

        outputs=outputs.view(-1,512)
        outputs=F.normalize(outputs, p=2, dim=1)

        return outputs

def extract(model):
    query='/home/afanti/LiChenggang/VehicleRI/VeRi/image_query'
    test='/home/afanti/LiChenggang/VehicleRI/VeRi/image_test'
    # a_feats=[]
    # b_feats=[]
    files_query=os.listdir(query)
    files_query.sort()
    files_test=os.listdir(test)
    files_test.sort()


    a_feats=torch.zeros(1678,512)
    b_feats=torch.zeros(11579,512)
    i=0
    j=0
    for f in files_query:
        img_path=os.path.join(query,f)
        img=Image.open(img_path).convert('RGB')
        # img=img.resize((80,160))
        img=img.resize((160,160))
        img=imgtransforms(img)
        #img=img.resize_(1,3,160,80)
        img=img.resize_(1,3,160,160)
        img=img.cuda()
        result=model(Variable(img)).data.cpu()
        a_feats[i]=result;
        i+=1
        # print result
        # exit()

    for f in files_test:
        img_path=os.path.join(test,f)
        img=Image.open(img_path).convert('RGB')
        # img=img.resize((80,160))
        img=img.resize((160,160))
        img=imgtransforms(img)
        #img=img.resize_(1,3,160,80)
        img=img.resize_(1,3,160,160)
        img=img.cuda()
        result=model(Variable(img)).data.cpu()
        b_feats[j]=result;

        j+=1

    print "a_feats:{} b_feats:{}".format(a_feats.size(),b_feats.size())
    rank1=compute(a_feats,b_feats,files_query,files_test)
    return rank1
    #return a_feats,b_feats


def compute(a_feats,b_feats,files_query,files_test):
    cmc=np.zeros(100,)
    #print "a_feats.size()[0]:",a_feats.size()[0]
    for i in xrange(a_feats.size()[0]):
        query_feat=a_feats[i].view(1,-1)
        query_feat=query_feat.expand(b_feats.size())
        dis=F.pairwise_distance(query_feat,b_feats,2)

        dis,order=torch.sort(dis,0)
        # for i in dis:
        #     print i
        
        order=order.numpy().reshape(-1,)
        # print order
        # a=np.where(order==i)[0][0]
        rank_flag = 0
        for a in xrange(1678):
            if (files_test[order[a]][0:4] == files_query[i][0:4]): #and (files_test[order[a]] != files_query[i]):
                rank_flag = a 
                #print "i: {} rank_flag: {}|".format(i,rank_flag),"files_query:{} files_test:{}".format(files_query[i],files_test[order[a]]) 
                break
        if a < 100:
            for j in xrange(rank_flag,100):
                cmc[j]+=1
        #break

    #print 'cmc[0]',cmc[0]*1.0/100
    return cmc




if __name__=='__main__':
    model=extract_model()
    model.load_state_dict(torch.load('/home/afanti/LiChenggang/VehicleRI/triplet/trip_res50_34999_batch_all.pkl',map_location={'cuda:1':'cuda:0'}))
    model.eval()
    model=model.cuda()
    rank1=extract(model)
    print rank1