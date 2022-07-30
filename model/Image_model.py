from distutils import extension
from turtle import forward, st
import torch
import torch.nn as nn
import numpy
import pdb
from torchvision import models

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant(m.bias, 0)
#ResNet
class Bottleneck(nn.Module):
    extension = 4
    def __init__(self, in_channel, mid_channel, stride, downsample=None):
        super(Bottleneck, self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.stride = stride
        self.downsample = downsample

        self.conv1 = nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1, stride=stride) #1x1
        self.bn1 = nn.BatchNorm2d(self.mid_channel)
        self.conv2 = nn.Conv2d(self.mid_channel, self.mid_channel,kernel_size=3,stride=1, padding=1) #3x3
        self.bn2 = nn.BatchNorm2d(self.mid_channel)
        self.conv3 = nn.Conv2d(self.mid_channel, self.mid_channel * self.extension, kernel_size=1,stride=1) #1x1
        self.bn3 = nn.BatchNorm2d(self.mid_channel * self.extension)

        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        if(self.downsample != None):
            residual = self.downsample(residual)
        
        out = residual + out
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.block = block
        self.layers = layers
        self.in_channel = 64

        #stem_layer
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #residual_layer
        self.stage1 = self.residual_layer(self.block, 64, self.layers[0], stride=1)
        self.stage2 = self.residual_layer(self.block, 128, self.layers[1], stride=2)
        self.stage3 = self.residual_layer(self.block, 256, self.layers[2], stride=2)
        self.stage4 = self.residual_layer(self.block, 512, self.layers[3], stride=2)

        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # self.fc1 = nn.Linear(2048, 1024)
        # self.fc_bn1 = nn.BatchNorm1d(1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc_bn2 = nn.BatchNorm1d(512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc_bn3 = nn.BatchNorm1d(256)

        # self.LeakyRelu = nn.LeakyReLU(0.1)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        B, _, _, _ = x.size()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out) #[B, 64, 55, 55]

        out = self.stage1(out)  #[B, 256, 55, 55]
        out = self.stage2(out)  #[B, 512, 28, 28]
        out = self.stage3(out)  #[B, 1024, 14, 14]
        out = self.stage4(out)  #[B, 2048, 7, 7]
        out = self.relu(self.bn2(self.conv2(out)))  #[B, 1024, 7, 7]

        out = self.avgpool(out)   #[B, 1024, 4, 4]

        out = out.view(B, 1024, -1)  #[B, 1024, 16]
        #pdb.set_trace()
        # out = self.LeakyRelu(self.fc_bn1(self.fc1(out)))
        # out = self.LeakyRelu(self.fc_bn2(self.fc2(out)))
        # out = self.LeakyRelu(self.fc_bn3(self.fc3(out)))
        #out = self.sigmoid(out)

        return out

    def residual_layer(self, block, mid_channel,block_num, stride=1):

        block_list = []

        downsample = None
        if(stride != 1 or self.in_channel != mid_channel*block.extension):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, mid_channel*block.extension, kernel_size=1, stride=stride),
                nn.BatchNorm2d(mid_channel*block.extension)
            )
        
        #Conv_Block
        conv_block = block(self.in_channel, mid_channel, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.in_channel = mid_channel*block.extension

        #Indentity Block
        for i in range(1,block_num):
            block_list.append(block(self.in_channel, mid_channel,stride=1))

        return nn.Sequential(*block_list)

if __name__ == '__main__':
    input = torch.randn(16, 3, 224, 224)
    Image_model = ResNet(Bottleneck,[3, 4, 6, 3])
    print(Image_model)
    Res50 = models.resnet50()
    print(Res50)
    # num = 0
    # for m in Image_model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         num += 1
    # print(num)
    # pdb.set_trace()
    # output = Image_model(input)

    # print(output.size())

