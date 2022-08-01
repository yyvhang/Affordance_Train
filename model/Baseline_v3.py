import numpy
import torch
import pdb
import torch.nn as nn
from Res50_basev3 import Res_50
from pointnet2_part import point_model



class model_encoder(nn.Module):
    def __init__(self, img_model_path, point_model_path, num_class = 18, pre_train = False):
        super().__init__() 
        self.img_encoder = Res_50()
        if pre_train:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = point_model(num_class)
        # point_model_dict = self.point_encoder.state_dict()
        # point_pretrain_dict = torch.load(point_model_path,map_location='cpu')
        # point_pretrain_dict = point_pretrain_dict['model_state_dict']
        # del point_pretrain_dict['sa1.mlp_convs.0.weight']
        # point_pretrain_dict={ k : v for k, v in point_pretrain_dict.items() if k in point_model_dict}
        # point_model_dict.update(point_pretrain_dict)
        # self.point_encoder.load_state_dict(point_model_dict)
        self.relu = nn.ReLU(inplace=True)
        self.fusion_Linear1 = nn.Linear(2048, 1024)
        self.fusion_bn1 = nn.BatchNorm1d(1024)
        self.fusion_Linear2 = nn.Linear(1024, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)

        self.fusion_conv1 = nn.Conv1d(128, 128, kernel_size=1, stride=1)
        self.fusion_conv2 = nn.Conv1d(128, num_class, kernel_size=1, stride=1)
        self.fusion_bn3 = nn.BatchNorm1d(num_class)

        self.sigmoid = nn.Sigmoid()
        



    def forward(self, img, points):
        B, C, N = points.size()
        img_out = self.img_encoder(img)                           #[B, 1024, 1]
        points_out, fusion_feature = self.point_encoder(points)   #[B, 128, 2048] --- [B, 1024, 1]
        Fusion = torch.cat([img_out, fusion_feature], dim=1)      #[B, 2048, 1]
        Fusion = Fusion.squeeze()                                 #[B, 2048]


        Fusion = self.relu(self.fusion_bn1(self.fusion_Linear1(Fusion)))  #[B, 1024]
        Fusion = self.relu(self.fusion_bn2(self.fusion_Linear2(Fusion)))  #[B, 128]
        Fusion = Fusion.view(B, 128, 1).repeat(1, 1, N)                   #[B, 128, 2048]

        out = Fusion + points_out
        out = self.relu(self.fusion_bn2(self.fusion_conv1(out)))          #[B, 128, 2048]
        out = self.sigmoid(self.fusion_bn3(self.fusion_conv2(out)))       #[B, 18, 2048]
        out = out.permute(0, 2, 1)                                        #[B, 2048, 18]

        return out

if __name__ == '__main__':
    img_pre_path = 'runs/pre_train/resnet50-19c8e357.pth'
    point_pre_path = 'runs/pre_train/pointnet2.pth'
    input_img = torch.randn(16, 3, 224, 224)
    input_point = torch.randn(16, 3, 2048)
    model = model_encoder(img_pre_path, point_pre_path)
    out = model(input_img, input_point)
    print(out.size())