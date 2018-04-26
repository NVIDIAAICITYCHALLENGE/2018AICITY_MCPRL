import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLayer(nn.Module):
    def __init__(self, model):
        super(TripletLayer, self).__init__()
        self.model=model


    def forward(self,embedded_anchor,embedded_positive,embedded_negative):
        self.embedded_anchor = self.model(embedded_anchor)
        self.embedded_positive=self.model(embedded_positive)
        self.embedded_negative=self.model(embedded_negative)
        dist_ap=torch.pow(F.pairwise_distance(self.embedded_anchor,self.embedded_positive,p=2),2)
        dist_an=torch.pow(F.pairwise_distance(self.embedded_anchor,self.embedded_negative,p=2),2)
        return dist_ap,dist_an#,self.embedded_anchor,embedded_positive,embedded_negative
