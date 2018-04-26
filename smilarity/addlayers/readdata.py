from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import torch
import torch.utils.data as data
import random


class readdatas():
    def __init__(self,batch_size,shape,max_per_id,filelist_path,transforms=transforms):

        self.batch_size=batch_size
        self.shape=shape
        self.h=self.shape[0]
        self.w=self.shape[1]
        self.filelist_path=filelist_path
        self.id_dict = self.readalldata(filelist_path)

        self.max_per_id = max_per_id
        self.all_id=self.id_dict.keys()
        random.shuffle(self.all_id)
        self.index=0
        self.transforms=transforms
        self.id_order=0




    def readalldata(self,filename):
        list_file=open(filename)
        content = list_file.read()
        lines = content.split('\n')
        all_id_dict={}
        for line in lines:
            if len(line.split())<2:
                continue
            file_name=line.split()[0]
            label_id=line.split()[1]
            if all_id_dict.has_key(label_id):
                all_id_dict[label_id].append(file_name)
            else:
                all_id_dict[label_id]=[file_name]
        return all_id_dict

    def sampledata(self):
        image_list=[]
        label_list=[]
        batch_list=self.get_batch_list()
        self.dataset=my_dataset(batch_list,train=True,transform=self.transforms,loader=self.loadimage)
        return self.dataset

    def get_batch_list(self):
        count=0

        while(count<self.batch_size):

            if self.id_order>=len(self.all_id):
                random.shuffle(self.all_id)
                self.id_order=0


            id=self.all_id[self.id_order]
            for file_name in self.id_dict[id][:self.max_per_id]:
                if count<self.batch_size:
                    count+=1
                    yield(file_name,id)
                else:
                    return
            self.id_order+=1


    def loadimage(self,filename):

        if not os.path.isfile(filename):
            print "Image file %s not exist!"%filename
            return None
        image=Image.open(filename).convert('RGB')
        image=image.resize((self.w,self.h))

        return image

class my_dataset(data.Dataset):
    def __init__(self,batch_list,loader,train=True,transform=None,target_transform=None):
        self.imgspath=[]
        self.labels=[]

        for file_name,label in batch_list:
            # print file_name,label
            self.imgspath.append(file_name)
            self.labels.append(int(label))

        self.transform=transform
        self.loader=loader

    def __getitem__(self,index):
        path=self.imgspath[index]
        label=self.labels[index]
        img=self.loader(path)
        if self.transform is not None:
            img=self.transform(img)
        return img,torch.LongTensor([label])
    def __len__(self):
        return len(self.labels)
