
import torch
import torch.nn as nn
from torchvision import models
import pdb
import torch.nn.functional as F


class Res_50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(pretrained=False)
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024)
        #self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, img):
        B, _, _, _ = img.size()

        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))
        out = self.model.maxpool(out)  #[B, 64, 56, 56]

        out = self.model.layer1(out)   #[B, 256, 56, 56]
        out = self.model.layer2(out)   #[B, 512, 28, 28]
        out = self.model.layer3(out)   #[B, 1024, 14, 14]
        out = self.model.layer4(out)   #[B, 2048, 7, 7]

        out = self.model.relu(self.bn2(self.conv2(out)))  #[B, 1024, 7, 7]
        #out = self.avgpool(out)   #[B, 1024, 4, 4]
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(B, 1024, -1)  #[B, 1024, 1]

        return out

if __name__=='__main__':
    img = torch.randn(16, 3, 224, 224)
    save_path = 'runs/pre_train/resnet50-19c8e357.pth'
    model = Res_50()
    out = model(img)
    print(out.size())
    #print(model)
    # out = model(img)
    # print(out.size())
    #print(model)
    # model_dict = model.state_dict()
    # pretrain_dict = torch.load(save_path)
    # for name in model_dict:
    #     print(name)
    # print('===============')
    # for name1 in list(pretrain_dict.keys()):
    #     new_key = 'model.' + name1
    #     pretrain_dict[new_key] = pretrain_dict.pop(name1)
    # pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in model_dict}
    # model_dict.update(pretrain_dict)
    # model.load_state_dict(model_dict)
    # out = model(img)
    # print(out.size())
    # for k,v in model_dict.items():
    #     print(f'{k} : {v}')
    # #print(pretrain_dict.keys())
    # print(model_dict.keys())
