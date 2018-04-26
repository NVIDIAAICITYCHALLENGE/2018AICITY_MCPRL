import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time

class l2_norm(nn.Module):
    def __init__(self, model):
        super(l2_norm, self).__init__()
        self.model=model


    def forward(self,inputs):

        self.feature=self.model(inputs)
        self.sqrt_sum=torch.sqrt(torch.sum(torch.pow(self.feature,2),1)).view(inputs.size()[0],1)
        self.sqrt_sum=self.sqrt_sum.expand(self.feature.size()[0],self.feature.size()[1])
        self.feature=self.feature/self.sqrt_sum
        return self.feature



class online_triplet_loss(nn.Module):
    def __init__(self,model):
        super(online_triplet_loss,self).__init__()
        self.model=model

    def forward(self,inputs,labels,margin):

        self.feature=self.model(inputs)
        features=F.normalize(self.feature, p=2, dim=1)


        num=labels.size()[0]
        m_label=labels.data.cpu().numpy()
        cmp=Variable(torch.Tensor([0]).cuda()).view(-1,1)
        count=0
        error=0

        criterion=nn.MarginRankingLoss(margin)

        distall={}
        #calculate pair distance
        for i in xrange(num):
            tmp_data=features[i].view(1,-1)
            tmp_data=tmp_data.expand(features.size())
            dist_all=torch.pow(F.pairwise_distance(tmp_data,features,2),2)
            distall[i]=dist_all


        prev=-1
        boundary=[]

        for i in xrange(m_label.shape[0]):
            if prev!=m_label[i][0]:
                boundary.append(m_label[i][0])
                prev=m_label[i][0]

        labels_dict={}

        for label in boundary:
            b=np.where(m_label==label)
            labels_dict[label]=[b[0][0],b[0][-1]]


        all_triplets_num=0
        nums_loss=0
        count_sum=0
        num_ap_an=0
        dist_an_all_end=Variable(torch.Tensor([[0]]).cuda())
        dist_ap_end=Variable(torch.Tensor([[0]]).cuda())
        for i in xrange(num):

            left_p,right_p=labels_dict[m_label[i][0]][0],labels_dict[m_label[i][0]][1]
            for j in xrange(left_p,right_p):
                count_sum+=1
                if j==i:continue
                dist_ap=distall[i][j]
                if left_p==0:dist_an_all=distall[i][right_p:]
                else:
                    if right_p==len(labels)-1:dist_an_all=distall[i][0:left_p]
                    else: dist_an_all=torch.cat((distall[i][0:left_p],distall[i][right_p:]),0)
                dist_ap=dist_ap.view(1,-1)
                dist_ap=dist_ap.expand(dist_an_all.size())
                #print dist_ap,dist_an_all.data
                num_ap_an+=torch.sum(torch.ge(dist_ap.data,dist_an_all.data))  #nums of ap>=an(error) to make ap<=an
                all_triplets_num+=dist_ap.size()[0]
                #batchloss+=criterion(dist_an_all,dist_ap,tmp)
                dist_an_all_end=torch.cat((dist_an_all_end,dist_an_all),0)
                dist_ap_end=torch.cat((dist_ap_end,dist_ap),0)

        dist_an_all_end=dist_an_all_end[1:]
        dist_ap_end=dist_ap_end[1:]
        target=Variable(torch.FloatTensor(dist_ap_end.size()).fill_(1).cuda())
        batchloss=criterion(dist_an_all_end,dist_ap_end,target)

        #batchloss=batchloss/all_triplets_num
        batchloss=batchloss
        accuracy=1-num_ap_an*1.0/all_triplets_num

        return batchloss ,accuracy


def triplet_loss_batch_all(features,labels,margin):
    num=labels.size()[0]
    m_label=labels.data.cpu().numpy()
    cmp=Variable(torch.Tensor([0]).cuda()).view(-1,1)
    count=0
    error=0

    criterion=nn.MarginRankingLoss(margin)


    distall=Variable(torch.zeros((num,num)).cuda())


    #calculate pair distance
    for i in xrange(num):
        tmp_data=features[i].view(1,-1)
        tmp_data=tmp_data.expand(features.size())
        dist_all=torch.pow(F.pairwise_distance(tmp_data,features,2),2)
        distall[i]=dist_all


    prev=-1
    boundary=[]

    for i in xrange(m_label.shape[0]):
        if prev!=m_label[i][0]:
            boundary.append(m_label[i][0])
            prev=m_label[i][0]

    labels_dict={}

    for label in boundary:
        b=np.where(m_label==label)
        labels_dict[label]=[b[0][0],b[0][-1]]


    all_triplets_num=0
    nums_loss=0
    count_sum=0
    num_ap_an=0
    dist_an_all_end=Variable(torch.Tensor([[0]]).cuda())
    dist_ap_end=Variable(torch.Tensor([[0]]).cuda())
    for i in xrange(num):

        left_p,right_p=labels_dict[m_label[i][0]][0],labels_dict[m_label[i][0]][1]
        for j in xrange(left_p,right_p+1):
            count_sum+=1
            if j==i:continue
            dist_ap=distall[i][j]
            if left_p==0:dist_an_all=distall[i][right_p:]
            else:
                if right_p==len(labels)-1:dist_an_all=distall[i][0:left_p]
                else: dist_an_all=torch.cat((distall[i][0:left_p],distall[i][right_p:]),0)
            dist_ap=dist_ap.view(-1,)

            dist_ap=dist_ap.expand(dist_an_all.size())

            num_ap_an+=torch.sum(torch.ge(dist_ap.data,dist_an_all.data))  #nums of ap>=an(error) to make ap<=an
            all_triplets_num+=dist_ap.size()[0]

            dist_an_all_end=torch.cat((dist_an_all_end,dist_an_all),0)
            dist_ap_end=torch.cat((dist_ap_end,dist_ap),0)

    dist_an_all_end=dist_an_all_end[1:]
    dist_ap_end=dist_ap_end[1:]

    target=Variable(torch.FloatTensor(dist_ap_end.size()).fill_(1).cuda())
    batchloss=criterion(dist_an_all_end,dist_ap_end,target)

    #batchloss=batchloss/all_triplets_num
    batchloss=batchloss
    accuracy=1-num_ap_an*1.0/all_triplets_num

    return batchloss ,accuracy


def triplet_loss_hard(features,labels,margin):

    features=F.normalize(features, p=2, dim=1)
    num=labels.size()[0]
    m_label=labels.data.cpu().numpy()
    cmp=Variable(torch.Tensor([0]).cuda()).view(-1,1)
    count=0
    error=0

    criterion=nn.MarginRankingLoss(margin)

    distall=Variable(torch.zeros((num,num)).cuda())
    #calculate pair distance
    for i in xrange(num):
        tmp_data=features[i].view(1,-1)
        tmp_data=tmp_data.expand(features.size())
        dist_all=torch.pow(F.pairwise_distance(tmp_data,features,2),2)

        distall[i]=dist_all


    prev=-1
    boundary=[]

    for i in xrange(m_label.shape[0]):
        if prev!=m_label[i][0]:
            boundary.append(m_label[i][0])
            prev=m_label[i][0]

    labels_dict={}

    for label in boundary:
        b=np.where(m_label==label)
        labels_dict[label]=[b[0][0],b[0][-1]]

    all_triplets_num=0
    nums_loss=0
    count_sum=0
    num_ap_an=0
    dist_an_all_end=Variable(torch.Tensor([[0]]).cuda())
    dist_ap_end=Variable(torch.Tensor([[0]]).cuda())

    for i in xrange(num):

        left_p,right_p=labels_dict[m_label[i][0]][0],labels_dict[m_label[i][0]][1]
        if left_p==right_p:
            continue
        if left_p==0:dist_an_all=distall[i][right_p+1:]
        else:
            if right_p==len(labels)-1:dist_an_all=distall[i][0:left_p]
            else: dist_an_all=torch.cat((distall[i][0:left_p],distall[i][right_p+1:]),0)
        dist_ap_all=distall[i][left_p:right_p+1]
        dist_ap=dist_ap_all.max()
        dist_an=dist_an_all.min()

        dist_an_all_end=torch.cat((dist_an_all_end,dist_an.view(-1,1)),0)

        dist_ap_end=torch.cat((dist_ap_end,dist_ap.view(-1,1)),0)


    dist_an_all_end=dist_an_all_end[1:]
    dist_ap_end=dist_ap_end[1:]
    target=Variable(torch.FloatTensor(dist_ap_end.size()).fill_(1).cuda())
    batchloss=criterion(dist_an_all_end,dist_ap_end,target)

    #accuracy=1-num_ap_an*1.0/all_triplets_num
    return batchloss#,accuracy







if __name__=="__main__":
    inputs=torch.Tensor([[2,2,2],[5,1,2],[4,3,1]])