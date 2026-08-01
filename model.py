


import torch
from torchvision import transforms

from dataset import YoloDataset
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16
class MymoduleforYolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_contract = vgg16(weights=VGG16_Weights.DEFAULT).features
    #我们要求最后 xcenter，ycenter，width，height，4个分类（摩托车呗），一共八个。
        self.fc_layer=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*16*16,2048),
            nn.ReLU(),
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,5)#加入了Confidence【x，y，w，h， con】//改进因为是单目标检测，去除 类别
            #是还是不是最后我要求（得到logit算概率）
        )
    def forward(self,x):
        x= self.img_contract(x)
        return self.fc_layer(x)

if __name__=="__main__":
    mymoudle=MymoduleforYolo()
    input=torch.rand((1,3,512,512))
    print(input)
    output=mymoudle(input)
    print(output.shape)
    print(mymoudle)
