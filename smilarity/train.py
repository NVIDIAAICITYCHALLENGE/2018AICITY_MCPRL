import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from addlayers.tripletnet import TripletLayer
import torch.nn.init as init
import numpy as np
import copy
import torchvision.models as models
from addlayers.mylayer import l2_norm
from addlayers import mylayer
from addlayers.readdata import readdatas
from addlayers.mylayer import online_triplet_loss
from addlayers import test
import time
import addlayers.model as my_model
import addlayers.mylayer1 as mylayer1


mytransforms=transforms.Compose([
    transforms.RandomCrop(size=(200,200)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ])

targetpath='/home/afanti/LiChenggang/VehicleRI/triplet'


def exp_lr_scheduler(optimizer,iters,iters_all=20000):
    lr=0.01*(0.1**(iters/iters_all))
    if iters % 20000==0:
        print 'LR is set to {}'.format(lr)
    for param in optimizer.param_groups:
        param['lr']=lr
    return optimizer




def train(model,my_data,optimizer,lr_scheduler,margin,iterations):

    loss=[]
    ranks=[]
    for iteration in xrange(iterations):
        optimizer=lr_scheduler(optimizer,iteration)
        dataset=my_data.sampledata()
        trainloader=torch.utils.data.DataLoader(dataset,batch_size=len(dataset),shuffle=False,num_workers=4)
        inputs,labels=iter(trainloader).next()

        inputs,labels=inputs.cuda(),labels.cuda()
        inputs,labels=Variable(inputs),Variable(labels)
        features=model(inputs)
        batchloss=mylayer1.triplet_loss_hard(features,labels,margin)
        optimizer.zero_grad()
        batchloss.backward()
        optimizer.step()

        if (iteration) % 20==0:

            print 'iteration {},loss {}'.format(iteration,batchloss.data[0])

        if (iteration+1) %1000==0 or iteration==0:
            torch.save(model.state_dict(),os.path.join(targetpath,'trip_alex_{}_hard.pkl'.format(iteration)))



if __name__=="__main__":
    torch.cuda.set_device(0)
    model=my_model.alex_trip()
    model=model.cuda()

    margin=0.2
    criterion=nn.MarginRankingLoss(margin)
    path='/home/afanti/LiChenggang/VehicleRI/triplet/train.txt'
    my_data=readdatas(300,(220,220),10,path,mytransforms)
    optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    train(model,my_data,optimizer,exp_lr_scheduler,margin,iterations=50000)
