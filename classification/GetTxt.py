# -*- coding: utf-8 -*-
import os, random
#make the image list for training and testing

rt = '../../data/vehicle/new_vehicle_3'
choises = ['vehicle', 'non_v']
v_folder = os.listdir(os.path.join(rt, choises[0]))
n_folder = os.listdir(os.path.join(rt, choises[1]))

l = []
for i in v_folder:
    l.append((os.path.join(rt, choises[0], i), 0))
for i in n_folder:
    l.append((os.path.join(rt, choises[1], i), 1))

random.shuffle(l)

test = l[:len(l)/9]
train = l[len(l)/9:]

f = open('train.txt', 'w')
for i in train:
    f.write(i[0]+' '+str(i[1])+'\n')
f.close()

f = open('test.txt', 'w')
for i in test:
    f.write(i[0]+' '+str(i[1])+'\n')
f.close()
