# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyLoader(Dataset):
    #DataLoader Class
    def __init__(self, txt, transforms=None, loader=default_loader):
        #txt is the path of training images
        self.loader = loader
        self.transforms = transforms
        f = open(txt, 'r')
        imgs = []
        for line in f:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        im, label = self.imgs[index]
        img = self.loader(im)
        #rotation
        rotation = random.randint(-10,10)
        img = img.rotate(rotation)
        target = int(label)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

if __name__ == '__main__':
    a = MyLoader('./train.txt')
    img, target = a.__getitem__(index=1)
    img.save(str(target)+'.jpg')
