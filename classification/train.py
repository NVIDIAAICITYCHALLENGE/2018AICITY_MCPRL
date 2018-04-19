# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import models
from MyLoader import MyLoader

import time, os
import numpy as np
from datetime import datetime

#Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Set hyperparameters for training model
lr = 0.0001
logdir = './logs'
decay_rate = 0.5
decay_step = 5000
experiment_name = 'vgg16'
extra_name = 'extra_data_8_2'
steps_to_show = 100
steps_to_save = 5000
batch_size = 128
num_cls = 2

def adjust_lr(optimizer, cur_stp, lr=lr, decay_rate=decay_rate, decay_step=decay_step):
    #Modify the learning rate of Optimizer in time, according the current step.
    new_lr = lr*(decay_rate**(cur_stp//decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr']=new_lr
    return new_lr

def accuracy(output, target):
    #Return the classify accuracy.
    #Output and target are like 2xN (where N is the Batch Size).
    _, tmp = torch.max(output.data, 1)
    return (tmp==target).sum()

def main():
    #The main() is used to finish all the training operation.

    #Record the training log using Tensorboard
    writer = SummaryWriter()

    #Load model
    net = models.Net(name=experiment_name)

    #Use CUDA or not
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()

    #Define the transform for loading image
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Scale(size=(64,64)),
                                transforms.ToTensor(),])

    #Set Adam as our optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

    #Load train and test data and set corresponding parameters
    TrainData = MyLoader('./train.txt', trans)
    trainloader = DataLoader(dataset=TrainData, batch_size=batch_size, num_workers=16, shuffle=True, )
    TestData = MyLoader('./test.txt', trans)
    testloader = DataLoader(dataset=TestData, batch_size=batch_size, num_workers=16, shuffle=True)

    #Set log path to save the training log.
    log_path = os.path.join(logdir, experiment_name+extra_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    #Petient is used to decide when to end training.
    step = 1
    duration = 0.
    precision = 0.
    petient = 0

    #Use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(1, 501):
        #Train
        for batch_idx, (img, target) in enumerate(trainloader):
            start_time = time.time()
            if use_cuda:
                img, target = img.cuda(), target.cuda()
            img, target = Variable(img), Variable(target)

            pred = net.train()(img)

            loss = criterion(pred, target)

            learning_rate = adjust_lr(optimizer=optimizer, cur_stp=step)

            #Show the information about training
            duration += time.time() - start_time
            if step%steps_to_show == 0:
                speed = batch_size*steps_to_show/duration
                duration = 0.
                print '=>%s: epoch: %d, step: %d, loss=%f, lr=%f, (%.1f examples/sec)' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, step, loss.data[0], learning_rate, speed)

            #Save model
            if step%steps_to_save==0:
                model_path = os.path.join(log_path, 'model_%d.tar'%(step))
                torch.save(net.state_dict(), model_path)
                print 'model saved at:', model_path

            writer.add_scalar('loss', loss.data[0], step)
            writer.add_scalar('lr', learning_rate, step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct_num = 0.
        count = 0

        #Evaluation
        for batch_idx, (img, target) in enumerate(testloader):
            if use_cuda:
                img, target = img.cuda(), target.cuda()
            img = Variable(img)
            pred = net.eval()(img)
            correct_num += accuracy(pred, target)
            count += target.size(0)

        #Decide whether to end training according to the precision and petient
        precision_ = float(correct_num)/float(count)
        if precision_ > precision:
            precision = precision_
            model_path = os.path.join(log_path, 'model_%.4f.tar'%(precision))
            torch.save(net.state_dict(), model_path)
            print 'accuracy: ', precision
            print 'model saved at:', model_path
        else:
            petient += 1
            if petient==100:
                print "Time to stop"
                writer.close()
                quit()
    writer.close()
    model_path = os.path.join(log_path, 'model_final.tar')
    torch.save(net.state_dict(), model_path)

if __name__ == '__main__':
    main()
