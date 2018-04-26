import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class alex_trip(nn.Module):
    def __init__(self):
        super(alex_trip,self).__init__()
        model=models.alexnet(pretrained=True)
        self.model1=model.features
        # self.fc1=nn.Linear(1024,4096)
        self.fc1=nn.Linear(6400,4096)
        self.fc2=model.classifier[2]
        self.dropout=nn.Dropout(0.5)
        self.relu=nn.ReLU()
        self.fc=nn.Linear(4096,512)
    def forward(self,inputs):
        #print 'inputs',inputs.size()
        inputs=self.model1(inputs)
        
        #print inputs.size()
        # inputs=inputs.view(-1,1024)
        inputs=inputs.view(-1,6400)
        inputs=self.dropout(self.relu(self.fc1(inputs)))
        inputs=self.dropout(self.relu(self.fc2(inputs)))
        outputs=self.fc(inputs)

        outputs=outputs.view(-1,512)
        outputs=F.normalize(outputs, p=2, dim=1)

        return outputs


class extract_model(nn.Module):
    def __init__(self):
        super(extract_model,self).__init__()

        model=models.alexnet(pretrained=True)
        self.model1=model.features
        self.fc1=nn.Linear(1024,4096)
        self.fc2=model.classifier[2]
        self.dropout=nn.Dropout(0.5)
        self.relu=nn.ReLU()
        self.fc=nn.Linear(4096,512)


    def forward(self,inputs):
        #print inputs.size()
        inputs=self.model1(inputs)
        inputs=inputs.view(-1,1024)
        inputs=self.relu(self.fc1(inputs))
        inputs=self.relu(self.fc2(inputs))
        outputs=self.fc(inputs)

        outputs=outputs.view(-1,512)
        outputs=F.normalize(outputs, p=2, dim=1)
        return outputs


class resnet_trip(nn.Module):
    def __init__(self):
        super(resnet_trip,self).__init__()
        model=models.resnet18(pretrained=True)
        self.model=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,
        model.layer2,model.layer3,model.layer4,nn.AvgPool2d(kernel_size=(5,3)))


    def forward(self,inputs):
        inputs=self.model(inputs)
        outputs=inputs.view(-1,512)
        outputs=F.normalize(outputs, p=2, dim=1)
        return outputs



class extract_model(nn.Module):
    def __init__(self):
        super(extract_model,self).__init__()
        model=models.resnet18(pretrained=False)
        self.model=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,
        model.layer2,model.layer3,model.layer4,nn.AvgPool2d(kernel_size=(5,3)))


    def forward(self,inputs):
        inputs=self.model(inputs)
        outputs=inputs.view(-1,512)
        outputs=F.normalize(outputs, p=2, dim=1)
        return outputs






class resnet50_trip(nn.Module):
    def __init__(self):
        super(resnet50_trip,self).__init__()
        model=models.resnet50(pretrained=True)
        # self.model=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,
        # model.layer2,model.layer3,model.layer4,nn.AvgPool2d(kernel_size=(5,3)))
        self.model=nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,
        model.layer2,model.layer3,model.layer4,nn.AvgPool2d(kernel_size=(5,5)))
        self.fc=nn.Linear(2048,512)

    def forward(self,inputs):
        inputs=self.model(inputs)

        outputs=inputs.view(-1,2048)
        outputs=self.fc(outputs)
        outputs=F.normalize(outputs, p=2, dim=1)

        return outputs

if __name__=="__main__":
    inputs=torch.Tensor(1,3,200,200);
    inputs=Variable(inputs.cuda())
    model=alex_trip().cuda()
    model(inputs)
