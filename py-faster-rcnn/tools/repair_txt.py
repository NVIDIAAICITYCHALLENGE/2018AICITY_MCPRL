# -*- coding: utf-8 -*-

#load the txt
with open('./txtfile/result_8_3*3_nclas.txt','r') as f1:
    lines = f1.readlines()
number,frame = 1,5
#write new txt
with open('./txtfile/result_8_3*3_nclas_re.txt','w') as f2:
    for line in lines:
        if int(line.split('/')[0]) != number:
            number = int(line.split('/')[0])
            frame = 5
        frame1 = int(line.split('/')[1].split('.')[0])
        if frame1 != frame :
            for i in range((frame1 - frame)):
                f2.write(str(number) + '/' + str(i + frame) + '.jpg ' + '\n')
            frame = frame1
        f2.write(line)
        frame += 5

