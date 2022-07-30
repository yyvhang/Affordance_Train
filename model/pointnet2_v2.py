import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import pdb

class Point_model(nn.Module):
    def __init__(self, num_classes):
        super(Point_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv_for_fusion = nn.Conv1d(512, 1024, 1)
        self.bn_fusion = nn.BatchNorm1d(1024)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv_2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = None    #[B, C, N]
        l0_xyz = xyz[:,:3,:]  #[B, 3, N]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 256, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  #[B, 512, 16]
        fusion_feature = l4_points
        fusion_feature = F.relu(self.bn_fusion(self.conv_for_fusion(fusion_feature))) #[B,1024,16]

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  #[B, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  #[B, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  #[B, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       #[B, 128, 2048]
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))     #[B, 128, 2048]
        x = self.conv_2(x)                                           #[B, num_class, 2048]
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x, fusion_feature


if __name__ == '__main__':
    import  torch
    model = Point_model(13)
    # xyz = torch.rand(6, 3, 2048)
    # pre_model_path = 'runs/pre_train/pointnet2.pth'
    # pretrain_dict = torch.load(pre_model_path,map_location='cpu')
    # pretrain_dict = pretrain_dict['model_state_dict']
    # del pretrain_dict['sa1.mlp_convs.0.weight']
    # model_dict = model.state_dict()
    # # for k in pretrain_dict:
    # #     print(f'{k}:{model_dict[k].size()}')
    # pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in model_dict}
    # model_dict.update(pretrain_dict)
    # model.load_state_dict(model_dict)
    # for k,v in model_dict.items():
    #     print(f'{k} : {v}')