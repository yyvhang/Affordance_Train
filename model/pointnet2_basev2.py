import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import pdb
import  torch

class Point_model(nn.Module):
    def __init__(self, num_classes):
        super(Point_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [128, 128, 256], False)   #[32, 32, 64]
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 256 + 3, [256, 256, 512], False) #[64, 64, 128],
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 512 + 3, [512, 512, 1024], False) #[128, 128, 256]
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 1024 + 3, [1024, 1024, 2048], False)  #[256, 256, 512]

        self.fp4 = PointNetFeaturePropagation(3072, [1024, 1024])  #768,256
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp2 = PointNetFeaturePropagation(768, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256, [128, 128, 128])

        self.conv_for_fusion = nn.Conv1d(2048, 1024, 1)
        self.bn_fusion = nn.BatchNorm1d(1024)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv_2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = None    #[B, C, N]
        l0_xyz = xyz[:,:3,:]  #[B, 3, N]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 256, 1024]   64
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 512, 256]    128
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 1024, 64]    256
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  #[B, 2048, 16]    512


        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  #[B, 1024, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  #[B, 512, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  #[B, 256, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       #[B, 128, 2048]

        feature_list = [l0_points, l1_points, l2_points, l3_points]
        return feature_list


if __name__ == '__main__':
    model = Point_model(18)
    input = torch.randn(16, 3, 2048)
    out = model(input)
    pdb.set_trace()
    print(len(out))