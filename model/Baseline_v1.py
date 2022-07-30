from turtle import forward
import numpy
import torch
import pdb
import torch.nn as nn
from model.ResNet50 import Res_50
from model.pointnet2_v2 import Point_model

class model_encoder(nn.Module):
    def __init__(self, img_model_path, point_model_path, num_class = 18):
        super().__init__() 
        self.img_encoder = Res_50()
        # pretrain_dict = torch.load(img_model_path)
        # img_model_dict = self.img_encoder.state_dict()
        # for k in list(pretrain_dict.keys()):
        #     new_key = 'model.' + k
        #     pretrain_dict[new_key] = pretrain_dict.pop(k)
        # pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in img_model_dict}
        # img_model_dict.update(pretrain_dict)
        # self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_model(num_class)
        # point_model_dict = self.point_encoder.state_dict()
        # point_pretrain_dict = torch.load(point_model_path,map_location='cpu')
        # point_pretrain_dict = point_pretrain_dict['model_state_dict']
        # del point_pretrain_dict['sa1.mlp_convs.0.weight']
        # point_pretrain_dict={ k : v for k, v in point_pretrain_dict.items() if k in point_model_dict}
        # point_model_dict.update(point_pretrain_dict)
        # self.point_encoder.load_state_dict(point_model_dict)

        self.fusion_conv1 = nn.Conv1d(16, num_class, 1)
        self.fusion_bn = nn.BatchNorm1d(num_class)
        self.relu = nn.ReLU()
        self.fusion_conv2 = nn.Conv1d(num_class, num_class, 1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, img, points):
        img_out = self.img_encoder(img)
        points_out, fusion_feature = self.point_encoder(points)
        Fusion_out = torch.cat([img_out, fusion_feature], dim=1)  #[B, 2048, 16]
        Fusion_out = Fusion_out.permute(0, 2, 1)
        out = self.relu(self.fusion_bn(self.fusion_conv1(Fusion_out)))

        out = out + points_out

        out = self.sigmoid(self.fusion_bn(self.fusion_conv2(out)))
        out = out.permute(0, 2, 1)

        return out

if __name__ == '__main__':
    img_pre_path = 'runs/pre_train/resnet50-19c8e357.pth'
    point_pre_path = 'runs/pre_train/pointnet2.pth'
    # input_img = torch.randn(16, 3, 224, 224)
    # input_point = torch.randn(16, 3, 2048)
    # model = model_encoder(img_pre_path, point_pre_path)
    # model_dict = model.state_dict()
    # for k,v in model_dict.items():
    #     print(f'{k} : {v}')
    # out = model(input_img, input_point)
    #print(out.size())