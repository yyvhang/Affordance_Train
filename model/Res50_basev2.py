import torch
import torch.nn as nn
from torchvision import models
import pdb

class Res_50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet50(pretrained=False)
        # self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
        # self.bn2 = nn.BatchNorm2d(1024)
        # self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, img):
        B, _, _, _ = img.size()

        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))
        out = self.model.maxpool(out)  #[B, 64, 56, 56]

        out1 = self.model.layer1(out)   #[B, 256, 56, 56]
        out2 = self.model.layer2(out1)   #[B, 512, 28, 28]
        out3 = self.model.layer3(out2)   #[B, 1024, 14, 14]
        out4 = self.model.layer4(out3)   #[B, 2048, 7, 7]

        out_list = [out1, out2, out3, out4]

        # out = self.model.relu(self.bn2(self.conv2(out)))  #[B, 1024, 7, 7]
        # out = self.avgpool(out)   #[B, 1024, 4, 4]
        # out = out.view(B, 1024, -1)  #[B, 1024, 16]

        return out_list

if __name__=='__main__':
    img = torch.randn(16, 3, 224, 224)
    save_path = 'runs/pre_train/resnet50-19c8e357.pth'
    model = Res_50()