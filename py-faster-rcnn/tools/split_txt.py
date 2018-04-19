# -*- coding: utf-8 -*-

f = open("2.txt")
out = open('train_blur.txt','w')
lines = f.readlines()
for line in lines:
    if line.split('/')[8] < "MVI_39801":
        out.write(line)
f.close()
out.close()
