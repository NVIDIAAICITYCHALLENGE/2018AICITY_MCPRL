# -*- coding: utf-8 -*-
import os

path,number = 1,1
data_path = '/home/sedlight/workspace/wei/data/CITYCHANLLENGE/all_imgs/bg/'
with open('test_bg_all.txt','w') as f:

    while os.path.exists(data_path + str(path)):
        image_path = data_path + str(path) + '/'
        while os.path.exists(image_path + str(number) +'.jpg'):
            f.write(str(path) + '/' + str(number) + '.jpg' + '\n')
            number += 1
        number = 1
        path += 1
