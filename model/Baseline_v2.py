import pdb
import torch
import torch.nn as nn
import numpy
from model.Fusion_Model import fusion_model
from model.pointnet2_basev2 import Point_model
from model.Res50_basev2 import Res_50


class baseline_v2(nn.Module):
    def __init__(self, img_prepath, num_class=18, pre_Train = True,
                 img_channel=[256, 512, 1024, 2048], point_channel=[128, 256, 512, 1024], mid_channel=[128, 256, 512, 1024]):
        super().__init__()
        self.img_channel = img_channel
        self.point_channel = point_channel
        self.mid_channel = mid_channel

        self.img_encoder = Res_50()
        if pre_Train:
            pretrain_dict = torch.load(img_prepath)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_model(num_class)
        self.relu = nn.ReLU()
        #nedd write
        self.img2d_to_3d_1 = nn.Conv2d(self.img_channel[0], self.img_channel[0], kernel_size=1, stride=1)
        self.bn_img3d_1 = nn.BatchNorm2d(self.img_channel[0])
        self.fusion_layer1 = fusion_model(self.img_channel[0], self.point_channel[0], self.mid_channel[0])

        self.img2d_to_3d_2 = nn.Conv2d(self.img_channel[1], self.img_channel[1], kernel_size=1, stride=1)
        self.bn_img3d_2 = nn.BatchNorm2d(self.img_channel[1])
        self.fusion_layer2 = fusion_model(self.img_channel[1], self.point_channel[1], self.mid_channel[1]) 
        
        self.img2d_to_3d_3 = nn.Conv2d(self.img_channel[2], self.img_channel[2], kernel_size=1, stride=1)
        self.bn_img3d_3 = nn.BatchNorm2d(self.img_channel[2])
        self.fusion_layer3 = fusion_model(self.img_channel[2], self.point_channel[2], self.mid_channel[2])

        self.img2d_to_3d_4 = nn.Conv2d(self.img_channel[3], self.img_channel[3], kernel_size=1, stride=1)
        self.bn_img3d_4 = nn.BatchNorm2d(self.img_channel[3])
        self.fusion_layer4 = fusion_model(self.img_channel[3], self.point_channel[3], self.mid_channel[3])  

        #FPN_layer
        self.fpn1 = nn.ConvTranspose1d(self.point_channel[3], self.point_channel[2], kernel_size=6, stride=4, padding=1)
        self.fpn_bn1 = nn.BatchNorm1d(self.point_channel[2])
        self.fpn2 = nn.ConvTranspose1d(self.point_channel[2], self.point_channel[1], kernel_size=6, stride=4, padding=1)
        self.fpn_bn2 = nn.BatchNorm1d(self.point_channel[1])
        self.fpn3 = nn.ConvTranspose1d(self.point_channel[1], self.point_channel[0], kernel_size=4, stride=2, padding=1)
        self.fpn_bn3 = nn.BatchNorm1d(self.point_channel[0])
        

        #predict_head
        self.head = nn.Conv1d(self.point_channel[0], num_class, kernel_size=1, stride=1)
        self.head_bn = nn.BatchNorm1d(num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, point):
        
        img_feature_list = self.img_encoder(img)
        point_feature_list = self.point_encoder(point)

        img3d_1 = self.relu(self.bn_img3d_1(self.img2d_to_3d_1(img_feature_list[0])))
        img3d_2 = self.relu(self.bn_img3d_2(self.img2d_to_3d_2(img_feature_list[1])))
        img3d_3 = self.relu(self.bn_img3d_3(self.img2d_to_3d_3(img_feature_list[2])))
        img3d_4 = self.relu(self.bn_img3d_4(self.img2d_to_3d_4(img_feature_list[3])))

        fusion_feature1 = self.fusion_layer1(point_feature_list[0], img_feature_list[0], img3d_1) #[B, 128, 2048]
        fusion_feature2 = self.fusion_layer2(point_feature_list[1], img_feature_list[1], img3d_2) #[B, 256, 1024]
        fusion_feature3 = self.fusion_layer3(point_feature_list[2], img_feature_list[2], img3d_3) #[B, 512, 256]
        fusion_feature4 = self.fusion_layer4(point_feature_list[3], img_feature_list[3], img3d_4) #[B, 1024, 64]
        fp1 = self.relu(self.fpn_bn1(self.fpn1(fusion_feature4) + fusion_feature3)) #[B, 512, 256]
        fp2 = self.relu(self.fpn_bn2(self.fpn2(fp1) + fusion_feature2))           #[B, 256, 1024]
        fp3 = self.relu(self.fpn_bn3(self.fpn3(fp2) + fusion_feature1))            #[B, 128, 2048]

        out = self.sigmoid(self.head_bn(self.head(fp3)))  #[B, 18, 2048]
        out = out.permute(0, 2, 1) #[B, 2048, 18]
        return out

if __name__=='__main__':
    img_input = torch.randn(16, 3, 224, 224)
    point_input = torch.randn(16, 3, 2048)
    model = baseline_v2(img_prepath='a', num_class=18, pre_Train=False)
    out = model(img_input, point_input)
    print(out.size())