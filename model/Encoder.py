import numpy
import torch
import pdb
import torch.nn as nn
from model.Image_model import Bottleneck, ResNet
from model.pointnet2_v2 import Point_model


class Fusion_model(nn.Module):
    def __init__(self, num_classes = 18):
        super(Fusion_model, self).__init__()

        self.img_encoder = ResNet(Bottleneck, [3, 4, 6, 3])

        self.point_encoder = Point_model(num_classes)
        self.fusion_conv1 = nn.Conv1d(16, num_classes, 1)
        self.fusion_bn = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()
        self.fusion_conv2 = nn.Conv1d(num_classes, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, points):

        img_out = self.img_encoder(img)  # [B, 1024, 16]
        points_out, fusion_feature = self.point_encoder(points)  # [B, 18, 2048], [B, 1024, 16]
        Fusion_out = torch.cat([img_out, fusion_feature], dim=1)  #[B, 2048, 16]
        Fusion_out = Fusion_out.permute(0, 2, 1)
        out = self.relu(self.fusion_bn(self.fusion_conv1(Fusion_out)))

        out = out + points_out

        out = self.sigmoid(self.fusion_bn(self.fusion_conv2(out)))
        out = out.permute(0, 2, 1)

        return out

if __name__ == '__main__':
    input_img = torch.randn(16, 3, 224, 224)
    input_points = torch.randn(16, 3, 2048)

    model = Fusion_model(num_classes=18)
    model_dict = model.state_dict()
    for name in model_dict:
        print(name)
    # if isinstance(model, nn.Conv2d):
    #     print('find!')
    #fusion_feature = model(input_img, input_points)





