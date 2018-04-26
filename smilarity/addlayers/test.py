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
    transforms.CenterCrop(size=(160,80)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    ])

class alex_trip(nn.Module):
    def __init__(self):
        super(alex_trip,self).__init__()
        model=models.alexnet(pretrained=True)
        self.model1=model.features
        self.fc1=nn.Linear(1024,4096)
        self.fc2=model.classifier[2]
        self.dropout=nn.Dropout(0.5)
        self.relu=nn.ReLU()
        self.fc=nn.Linear(4096,512)
    def forward(self,inputs):
        #print 'inputs',inputs.size()
        inputs=self.model1(inputs)
        #print inputs.size()
        inputs=inputs.view(-1,1024)
        inputs=self.dropout(self.relu(self.fc1(inputs)))
        inputs=self.dropout(self.relu(self.fc2(inputs)))
        outputs=self.fc(inputs)

        outputs=outputs.view(-1,512)
        outputs=F.normalize(outputs, p=2, dim=1)

        return outputs

class extract_model(nn.Module):
    def __init__(self):
        super(extract_model,self).__init__()
        model=models.resnet18(pretrained=True)
        self.model=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,
        model.layer2,model.layer3,model.layer4,nn.AvgPool2d(kernel_size=(5,3)))


    def forward(self,inputs):
        inputs=self.model(inputs)
        outputs=inputs.view(-1,512)
        outputs=F.normalize(outputs, p=2, dim=1)
        return outputs

def ComputeEuclid(feat_a,feat_b):
    feat_diff = feat_a-feat_b
    score = sum(feat_diff*feat_diff)
    return score


#def GetRanks(a1_feats,a2_feats,b1_feats,b2_feats,idx,idx2):
def GetRanks(a1_feats,b1_feats,idx,idx2):
    # Load image a feat
    feat_a1 = a1_feats[idx][idx2,:]
    #feat_a2 = a2_feats[idx][idx2,:]

    tmp_ranks=[]
    for n in xrange(0,cand_num):
        for n2 in xrange(0,b1_feats[n].shape[0]):
            feat_b1 = b1_feats[n][n2,:]
            #feat_b2 = b2_feats[n][n2,:]
            score= ComputeEuclid(feat_a1,feat_b1)
            tmp_ranks.append((n,n2,score))
    tmp_ranks=np.vstack(tmp_ranks)
    #rank
    idx_sort=np.argsort(tmp_ranks[:,2])
    tmp_ranks=tmp_ranks[idx_sort,:]
    best_rank=-1
    for i in xrange(0,tmp_ranks.shape[0]):
        if(idx==tmp_ranks[i,0]):
            best_rank=i+1
            break

    #print 'ID %d image %d best rank is %d.' %(idx,idx2,best_rank)
    return best_rank


def GetRank(a1_feats,b1_feats,idx):
    # Load image a feat
    feat_a1 = a1_feats[idx]
    feat_a1=feat_a1.reshape(512,)
    tmp_ranks=[]
    for n in xrange(0,100):

        feat_b1 = b1_feats[n]
        feat_b1=feat_b1.reshape(512,)
        score = ComputeEuclid(feat_a1,feat_b1)
        tmp_ranks.append((n,score))


    tmp_ranks=np.vstack(tmp_ranks)
    dist=tmp_ranks[:,1]



    #rank
    idx_sort=np.argsort(tmp_ranks[:,1])
    tmp_ranks=tmp_ranks[idx_sort,:]
    best_rank=-1
    for i in xrange(0,tmp_ranks.shape[0]):
        if(idx==tmp_ranks[i,0]):
            best_rank=i+1
            break

    #print 'ID %d image %d best rank is %d.' %(idx,idx2,best_rank)
    return best_rank,dist


def test(model):
    indir='/home/mmc/feiwei/dataset/cuhk03/query1'

    a_feats=[]
    b_feats=[]
    files=os.listdir(indir)
    files.sort()

    for f in files:
        img_path='{}/{}'.format(indir,f)
        img=Image.open(img_path).convert('RGB')
        img=img.resize((80,160))
        img=imgtransforms(img)
        img=img.resize_(1,3,160,80)
        img=img.cuda()
        result=model(Variable(img))

        feat=result.data.cpu().numpy()
        tmp_feat =  copy.deepcopy(feat)
        tmp_feat=tmp_feat.reshape(512,)
        if f.split('_')[1]=='00':
            a_feats.append(tmp_feat)
        else:
            b_feats.append(tmp_feat)


    a_feats= np.vstack(a_feats)
    b_feats= np.vstack(b_feats)

    ranks=[]
    dist=[]
    for idx in xrange(0,100):

        tmp_rank,dis=GetRank(a_feats,b_feats,idx)
        ranks.append(tmp_rank)
        dist.append(dis)

    for idx in xrange(0,100):

        tmp_rank,dis=GetRank(b_feats,a_feats,idx)
        ranks.append(tmp_rank)
        dist.append(dis)

    ranks=np.vstack(ranks)
    dist=np.vstack(dist)


    sort=np.argsort(dist,1)

    msort=np.zeros(sort.shape)
    for i in xrange(sort.shape[0]):
        for j in xrange(sort[0].shape[0]):
            msort[i][j]=np.where(sort[i]==j)[0]
    msort=msort+1

    ranks=[]

    for i in xrange(sort.shape[0]):
        #print np.where(sort[i]==i%316)[0][0]
        ranks.append(np.where(sort[i]==i%100)[0][0])
    ranks=np.vstack(ranks)+1

    for k in xrange(0,1):
        if(k==0):
            rank_thrd=1
        else:
            rank_thrd=k*5
        count=0
        #for i in xrange(ranks.shape[0]/2,ranks.shape[0]):
        for i in xrange(0,ranks.shape[0]):
            if(ranks[i]<=rank_thrd):
                count += 1
        accuracy=float(count)/(ranks.shape[0])
        print 'Rank%d accuray=%f' %(rank_thrd,accuracy)


    #reranking
    new_dist=np.zeros(shape=dist.shape)
    for i in xrange(dist.shape[0]/2):
        for j in xrange(dist[i].shape[0]):
            probe_dist=dist[i][j]

            new_dist[i][j]=dist[i][j]*(1+(1-np.exp(-((np.where(dist[j+100]<probe_dist)[0].shape[0])))))
            #new_dist[i][j]=dist[i][j]*(np.where(dist[j+100]<probe_dist)[0].shape[0]+(np.where(sort[i]==j)[0][0]))

    for m in xrange(dist.shape[0]/2,dist.shape[0]):
        for n in xrange(dist[m].shape[0]):
            probe_dist=dist[m][n]
            new_dist[m][n]=dist[m][n]*(1+(1-np.exp(-((np.where(dist[n]<probe_dist)[0].shape[0])))))
            #new_dist[m][n]=dist[m][n]*(np.where(dist[j+100]<probe_dist)[0].shape[0]+(np.where(sort[m]==n)[0][0]))
    newsort=np.argsort(new_dist,1)
    ranks=[]

    for i in xrange(newsort.shape[0]):

        ranks.append(np.where(newsort[i]==i%100)[0][0])
    ranks=np.vstack(ranks)+1


def extract(model):
    indir='/home/mmc/feiwei/dataset/cuhk03/query1'

    # a_feats=[]
    # b_feats=[]
    files=os.listdir(indir)
    files.sort()
    a_feats=torch.zeros(100,512)
    b_feats=torch.zeros(100,512)
    i=0
    for f in files:
        img_path=os.path.join(indir,f)
        img=Image.open(img_path).convert('RGB')
        # img=img.resize((80,160))
        img=img.resize((100,200))
        img=imgtransforms(img)
        #img=img.resize_(1,3,160,80)
        img=img.resize_(1,3,160,80)
        img=img.cuda()
        result=model(Variable(img)).data.cpu()

        if f.split('_')[1]=='00':
            a_feats[i/2]=result
        else:
            b_feats[i/2]=result
        i+=1
    rank1=compute(a_feats,b_feats)
    return rank1
    #return a_feats,b_feats


def compute(a_feats,b_feats):
    cmc=np.zeros(100,)
    for i in xrange(a_feats.size()[0]):
        query_feat=a_feats[i].view(1,-1)
        query_feat=query_feat.expand(a_feats.size())
        dis=F.pairwise_distance(query_feat,b_feats,2)

        dis,order=torch.sort(dis,0)
        order=order.numpy().reshape(-1,)

        a=np.where(order==i)[0][0]

        for j in xrange(a,100):
            cmc[j]+=1

    #print 'cmc[0]',cmc[0]*1.0/100
    return cmc[0]*1.0/100




if __name__=='__main__':
    # model=extract_model()
    model=alex_trip()
    #model.load_state_dict(torch.load('/home/mmc/feiwei/triplet-pytorch/models/res18_trip/trip_res18_cuhk03_19999.pkl'))
    model.load_state_dict(torch.load('/home/mmc/feiwei/triplet-pytorch/models/alex_trip/trip_alex_cuhk03_39999.pkl'))
    model.eval()
    model=model.cuda()
    #test(model)
    rank1=extract(model)
    print rank1
    #compute(a_feats,b_feats)
