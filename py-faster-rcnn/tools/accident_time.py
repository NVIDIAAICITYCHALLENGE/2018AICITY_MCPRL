# -*- coding: utf-8 -*-
from car_faster import faster
import _init_paths
import cv2 as cv
import numpy as np
import caffe,os
from fast_rcnn.config import cfg
#from car_faster import faster
from similar import GetSimilar

#the classes
CLASSES = ('__background__',
           'car')

#the detect model name
NETS = {'vgg_cnn_m_1024':('VGG_CNN_M_1024','vgg_cnn_m_1024_faster_rcnn_iter_100000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_blur_iter_10000.caffemodel','vgg_cnn_m_1024_faster_rcnn_blur_train_iter_20000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_truck_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_dark_iter_50000.caffemodel','vgg_cnn_m_1024_faster_rcnn_all1_iter_50000.caffemodel',
                          'vgg_cnn_m_1024_faster_rcnn_alldark_iter_50000.caffemodel'),
    'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_80000.caffemodel','vgg16_faster_rcnn_dark_iter_50000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

#compute similarity with resnet50
def sim(im_name1,im_name2,box1,box2,s):
    img1 = cv.imread('../data/track2_bg/' + im_name1)
    img2 = cv.imread('../data/track2_bg/' + im_name2)
    img1 = img1[int(float(box1[1])):int(float(box1[3])),int(float(box1[0])):int(float(box1[2]))]
    img2 = img2[int(float(box2[1])):int(float(box2[3])),int(float(box2[0])):int(float(box2[2]))]
    return s.similar(img1,img2)

def nms(label):
    if len(label) == 0:
        return []
    label = sorted(label,key = lambda l:l[-1],reverse = True)
    b = 0
    exsit,result = [],[]
    exsit.append(label[0])
    start,end = 1, len(label) - 1
    while start <= len(label) - 1:
        b = 0
        for i in exsit:
            bbox = {'0':[i[0],i[1],i[2],i[3]]}
            if isoverlap(label[start][:4],bbox,0.8)[0] != -1:
                b = 1
                break
        if b == 0:
            exsit.append(label[start])
        start += 1
    return exsit

#compute the start frame
def search_back(number,frame,bbox,detect,net):
    back_frame = int(frame) - 5  #detect every fifth previous frame
    miss,start,label = 0,int(frame),[]
    w,h = 7*(float(bbox[2]) - float(bbox[0])),7*(float(bbox[3]) - float(bbox[1]))
    x,y = float(bbox[0]) - 3*(float(bbox[2]) - float(bbox[0])),float(bbox[1]) - 3*(float(bbox[3]) - float(bbox[1]))
    scale = [max(x,0),max(y,0),min(x + w,800),min(y + h,410)] #the detection area
    cx,cy = (scale[0] + scale[2])/2,(scale[1] + scale[3])/2
    #multi-scale
    wnd1 = [max(cx-225,0) - max(cx + 225-800,0),max(cy - 127.5,0) - max(cy + 127.5-410,0),min(cx + 255,800) - min(cx - 255,0),min(cy + 127.5,410) - min(cy - 127.5,0)]
    wnd2 = [max(cx-150,0) - max(cx + 150-800,0),max(cy - 85,0) - max(cy + 85 - 410,0),min(cx + 150,800) - min(cx - 150,0),min(cy + 85,410) - min(cy - 85,0)]
    #if not a same car in the area continuously 6 times, break
    while back_frame > 0 and miss < 6:
        label,bbox2 = [],[]
        im = cv.imread('../data/track2_all/' + number + '/' + str(back_frame) + '.jpg')
        #multi-scale detection
        im1 = im[int(wnd1[1]):int(wnd1[3]),int(wnd1[0]):int(wnd1[2])]
        im2 = im[int(wnd2[1]):int(wnd2[3]),int(wnd2[0]):int(wnd2[2])]
        im1 = cv.resize(im1,(800,410))
        im2 = cv.resize(im2,(800,410))
        dets = detect.demo(net,im)
        label.extend(detect.bbox_get1(dets,0.8))
        dets = detect.demo(net,im1)
        for i in detect.bbox_get1(dets,0.8):
            label.append([i[0]*450/800.0 + wnd1[0],i[1]*255/410.0 + wnd1[1],i[2]*450/800.0 + wnd1[0],i[3]*255/410.0 + wnd1[1],i[4]])
        dets = detect.demo(net,im2)
        for i in detect.bbox_get1(dets,0.8):
            label.append([i[0]*300/800.0 + wnd2[0],i[1]*170/410.0 + wnd2[1],i[2]*300/800.0 + wnd2[0],i[3]*170/410.0 + wnd2[1],i[4]])
        #load frame
        points,max_over = 9,0.3
        for i in nms(label):
            bbox1 = {'0':[i[0],i[1],i[2],i[3]]}
            #judge if in the area
            if isoverlap(scale,bbox1,0)[0] != -1 and detect.classification(im,i):
                #judge if overlap with the anormally
                a,b = isoverlap(bbox[:4],bbox1,0.3)
                #select the most iou
                if b > max_over:
                    start = back_frame #refresh the start frame
                    miss = -1
                    max_over = b
                    bbox2 = i[:] #refresh the position of anormally
        bbox = bbox2[:] if len(bbox2) else bbox
        miss += 1
        back_frame -= 5
    return start

#compute the surf feature
def surf(im_name1,im_name2,box1,box2):
    im1 = cv.imread('../data/track2_bg/' + im_name1)
    im2 = cv.imread('../data/track2_bg/' + im_name2)
    #add some background
    im1 = im1[max(int(float(box1[1]) - 10),0):min(int(float(box1[3]) + 10),410),max(int(float(box1[0]) - 10),0):min(int(float(box1[2]) + 10),800)]
    im2 = im2[max(int(float(box2[1]) - 10),0):min(int(float(box2[3]) + 10),410),max(int(float(box2[0]) - 10),0):min(int(float(box2[2]) + 10),800)]
    #resize on same scale
    im1 = cv.resize(im1,(256,256))
    im2 = cv.resize(im2,(256,256))
    ds1,ds2 = [],[]
    surf = cv.xfeatures2d.SURF_create(300)
    kp1,ds1 = surf.detectAndCompute(im1,None)
    kp2,ds2 = surf.detectAndCompute(im2,None)

    index_params = dict(algorithm = 0,trees = 5)
    search_params = dict(checks = 50)
    matches = []
    flann = cv.FlannBasedMatcher(index_params,search_params)
    try:
        if len(ds1) >= 2 and len(ds2) >= 2:
            matches = flann.knnMatch(ds1,ds2,k = 2)
    except TypeError:
        return 0,1000
    else:
        good = []
        for m,n in matches:
            #compare the distance of the nearest and second nearest
            if m.distance < 0.8*n.distance:
                good.append(m)
        return len(good),min(len(ds1),len(ds2))

#compute the sift feature
def sift(im_name1,im_name2,box1,box2):
    im1 = cv.imread('../data/track2_bg/' + im_name1)
    im2 = cv.imread('../data/track2_bg/' + im_name2)
    im1 = im1[int(float(box1[1])):int(float(box1[3])),int(float(box1[0])):int(float(box1[2]))]
    im2 = im2[int(float(box2[1])):int(float(box2[3])),int(float(box2[0])):int(float(box2[2]))]
    im1 = cv.resize(im1,(128,128))
    im2 = cv.resize(im2,(128,128))
    ds1,ds2 = [],[]
    sift = cv.xfeatures2d.SIFT_create()
    kp1,ds1 = sift.detectAndCompute(im1,None)
    kp2,ds2 = sift.detectAndCompute(im2,None)

    print(im_name1,im_name2)
    index_params = dict(algorithm = 0,trees = 5)
    search_params = dict(checks = 50)
    matches = []
    flann = cv.FlannBasedMatcher(index_params,search_params)
    try:
        if len(ds1) >= 2 and len(ds2) >= 2:
            matches = flann.knnMatch(ds1,ds2,k = 2)
    except TypeError:
        return 0
    else:
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)
        return len(good)

#compute the ssim feature
def ssim(im_name1,im_name2,box1,box2):
    im1 = cv.imread('../data/track2-bg-imgs/' + im_name1)
    im2 = cv.imread('../data/track2-bg-imgs/' + im_name2)
    print(im_name2)
    print(im_name1)
    im1 = im1[int(float(box1[1])):int(float(box1[3])),int(float(box1[0])):int(float(box1[2]))]
    im2 = im2[int(float(box2[1])):int(float(box2[3])),int(float(box2[0])):int(float(box2[2]))]
    im1 = cv.resize(im1,(64,64))
    im2 = cv.resize(im2,(64,64))
    c1,c2 = 6.5025,58.5225
    img1,img2 = np.float32(im1),np.float32(im2)
    im11,im22,im12 = img1*img1,img2*img2,img1*img2
    mu1,mu2 = cv.GaussianBlur(img1,(11,11),1.5),cv.GaussianBlur(img2,(11,11),1.5)
    mu11,mu22,mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    sig11,sig22,sig12 = cv.GaussianBlur(im11,(11,11),1.5) - mu11, cv.GaussianBlur(im22,(11,11),1.5) - mu22,cv.GaussianBlur(im12,(11,11),1.5) - mu12
    t1,t2 = (2*mu12 + c1)*(2*sig12 + c2),(mu11 + mu22 + c1)*(sig11 + sig22 + c2)
    s = t1/t2
    s = sum(cv.mean(s))/3
    print(s)
    return s

#compute the iou,and return the largest one
def isoverlap(bbox,car1,thresh = 0.3):
    [x1,y1,x2,y2] = bbox
    a,b = -1,thresh
    for j in car1:
        [cx1,cy1,cx2,cy2] = car1[j][:4]
        if not(float(x1) > float(cx2) or float(x2) < float(cx1) or float(y1) > float(cy2) or float(y2) < float(cy1)):
            [sx1,sx2,sx3,sx4] = sorted([float(x1),float(x2),float(cx1),float(cx2)])
            [sy1,sy2,sy3,sy4] = sorted([float(y1),float(y2),float(cy1),float(cy2)])
            s1 = (float(sx3) - float(sx2))*(float(sy3) - float(sy2))
            s2 = (float(x2) - float(x1))*(float(y2) - float(y1)) + (float(cx2) - float(cx1))*(float(cy2) - float(cy1)) - s1
            if s1/s2 > b:
                a,b = j,s1/s2
    return a,b

time,final_score = {},[]
#set gpu
caffe.set_mode_gpu()
caffe.set_device(0)
cfg.GPU_ID = 0
#set the xml path and image path
save_path = os.path.join('../out/')
image_path = os.path.join('../data/track2_all/')
#load detection model
car_det = faster(save_path,image_path,0)

args = car_det.parse_args()

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
cfg.TEST.SCALES = (410,)
prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.DATA_DIR,
                              NETS[args.demo_net][2])

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

print ('\n\nLoaded network {:s}'.format(caffemodel))
#load Resnet50 model
si = GetSimilar()
#read the txt contain the bbox information
with open('txtfile/result_8_3*3_100_all_clas_re.txt','r') as f:
    lines = f.readlines()
car = {'0':[0,0,0,0,0,0,0,0,[]]} #{frame:[x1,y1,x2,y2,score,total frames,total apperance times,end frame,end position]}
number = lines[0].split('/')[0] #the number of video
for line in lines:
    #we compute the start time, after loading a video
    #if a car's total frames larger the 30s and total apperance times lager than 25s, save it in 'time'
    if line.split('/')[0] != number:
        for j in car:
            if float(car[j][6]) > 150 and int(car[j][7]) - int(j) > 900:
                score = float(car[j][6])/(float((car[j][7]) - float(j))/5 + 1)
                k = number + ' ' + str(int(j))
                if k not in time:
                    time[k] = [number,str(int(j)),score,car[j][8],car[j][7]]
                elif score > time[k][-3]:
                    time[k][-3] = score
        car = {'0':[0,0,0,0,0,0,0,0,[]]}
        number = line.split('/')[0]
    num = int((len(line.split(' ')) - 2) / 5)
    new_car,c,delete = [],0,[]
    frame = line.split(' ')[0].split('/')[1].split('.')[0]

    for i in range(num):
        a = 0
        im_name1 = line.strip('\n').split(' ')[0]

        bbox = line.split(' ')[1 + 5*i:6 + 5*i]
        a,_ = isoverlap(bbox[:4],car)
        #if not overlap save in new_car
        if a == -1:
            new_car.append(i)
        else:
            #judge the time interval
            if int(frame) - car[a][7] > 299:
                im_name2 = line.split('/')[0] + '/' + str(car[a][7]) + '.jpg'
                #good,max_good = surf(im_name1,im_name2,bbox[:4],car[a][:4])
                #if the distance of similarity < 0.9, we persume they are same cars
                c = sim(im_name1,im_name2,bbox[:4],car[a][:4],si)
                #if good > 9:
                if c < 0.9:
                    car[a][:4] = bbox[:4]
                    car[a][6] += 1
                    car[a][4] = float(car[a][4]) + float(bbox[4])
                    car[a][7] = int(frame)
                else: #else delete it
                    delete.append(a)
                    car['0'] = [bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],0,1,int(frame),[0,0,0,0]]
            else:
                car[a][:4] = bbox[:4]
                car[a][6] += 1
                car[a][7] = int(frame)
                car[a][4] = float(car[a][4]) + float(bbox[4])
    #if not overlapping, compute the similarity with the anormally appered in 20s
    for i in new_car:
        bbox = line.split(' ')[1+5*i:6+5*i]
        d,f = 0.9,0
        for j in car:
            if j != '0' and  int(frame) - car[j][7] < 600:
                im_name2 = line.split('/')[0] + '/' + str(int(car[j][7])) + '.jpg'
                #good,max_good = surf(im_name1,im_name2,bbox[:4],car[j][:4])
                c = sim(im_name1,im_name2,bbox[:4],car[j][:4],si)
                #if good > d:
                if c < d:
                    d,f = c,j
        #if all the distance > 0.9, it is a new anormally
        if d == 0.9:
            car[frame] = [bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],0,1,int(frame),bbox]
        else:
            car['0' + f] = [bbox[0],bbox[1],bbox[2],bbox[3],car[f][4],car[f][5],car[f][6],int(frame),car[f][8]]
    for j in car:
        car[j][5] += 1
    for d in delete:
        print('delete',d)
        if d in car:
            del car[d]
    delete = []

for j in car:
    if car[j][6] > 150 and int(car[j][7]) - int(j) > 900 :
        score = float(car[j][6])/(float((car[j][7]) - float(j))/5 + 1)
        k = number + ' ' + str(int(j))
        if k not in time:
            print j
            time[k] = [number,str(int(j)),score,car[j][8],car[j][7]]
        elif score > time[k][-3]:
            time[k][-3] = score
time_re = []
with open('start_time3.txt','w') as f1:
    for i in time:
        a = 0
        if time[i][2] > 0.3: #if the score > 0.3, we persume it is an anormally
            back_frame = search_back(time[i][0],time[i][1],time[i][3],car_det,net)
            time_re.append((time[i][0], back_frame, time[i][2],time[i][4]))
            print(time[i][0],float(back_frame)/30.0,time[i][2],time[i][3])
    print 'start'
    time_re.sort(key=lambda x: x[0])
    print time_re
    m = 0
    i,name,result = 0,time_re[0][0],[]
    while i < len(time_re):
        if time_re[i][0] == name and i != len(time_re) - 1:
            result.append(time_re[i])
        else:
            if i == len(time_re) - 1 and name == time_re[i][0]:
                result.append(time_re[i])
            elif i == len(time_re) - 1 and name !=time_re[i][0]:
                m = 1
            name = time_re[i][0]
            result.sort(key = lambda x:x[1])
            print (result[0][0],float(result[0][1])/30.0,result[0][2])
            #f1.write((str(result[0][0]) +' {:.2f} {:.2f}'  ).format(float(result[0][1])/30.0,result[0][2]))
            #f1.write('\n')
            #judge the interval is larger than 2 mins
            for n,j in enumerate(result[1:]):
                if int(j[1]) - int(result[n][1]) > 3600 and int(j[1]) - int(result[n][-1]) > 3600:
                    print(j[0],float(j[1])/30.0,j[2])
                    #f1.write((str(j[0]) +' {:.2f} {:.2f}'  ).format(float(j[1])/30.0,j[2]) + '\n')
            result = [time_re[i]]
        i += 1
    if m == 1:
        print (time_re[-1][0],float(time_re[-1][1])/30.0,time_re[-1][2])
        #f1.write((str(time_re[-1][0]) +' {:.2f} {:.2f}'  ).format(float(time_re[-1][1])/30.0,time_re[-1][2]))

