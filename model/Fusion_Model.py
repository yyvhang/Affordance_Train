
import torch
import torch.nn as nn
import pdb

class fusion_model(nn.Module):
    def __init__(self, img_channel, point_channel, mid_channel):
        super().__init__()
        self.mid_channel = mid_channel
        self.img_channel = img_channel
        self.p_channel = point_channel

        self.img_mlp = nn.Linear(self.img_channel, self.mid_channel)
        self.bn1 = nn.BatchNorm1d(self.mid_channel)
        self.point_mlp = nn.Linear(self.p_channel, self.mid_channel)
        self.fusion_mlp = nn.Linear(self.p_channel + self.mid_channel, self.p_channel)
        self.bn2 = nn.BatchNorm1d(self.p_channel)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, F_point, F_img2d, F_img3d):
        '''
        F_point : [B, C, N_p]
        F_img2d, F_img3d : [B, C, H, W]
        '''

        #fusion with img2d
        F_point = F_point.permute(0, 2, 1)  #[B, N_p, C]
        point = self.point_mlp(F_point)
        point = self.bn1(point.permute(0, 2, 1))   #[B, C1, N_p]

        B,C = F_img2d.shape[0], F_img2d.shape[1]
        F_img2d = F_img2d.contiguous().view(B, C, -1)
        F_img2d = F_img2d.permute(0, 2, 1)  #[B, N_i, C]
        img = self.img_mlp(F_img2d)
        img = self.bn1(img.permute(0, 2, 1))   #[B, C1, N_i]
        scale = img.shape[1] ** (-0.5)
        point_T = point.permute(0, 2, 1)  #[B, N_p, C1]
        atten_1 = torch.bmm(point_T, img) * scale#[B, N_p, N_i]
        atten_1 = self.softmax(atten_1)
        atten_out_1 = torch.bmm(atten_1, img.permute(0, 2, 1))  #[B, N_p, C1]
        fusion_feature1 = torch.cat((atten_out_1, F_point), dim=2)  #[B, N_p, C1+C]
        fusion_feature1 = self.fusion_mlp(fusion_feature1)
        fusion_feature1 = self.bn2(fusion_feature1.permute(0, 2, 1)) #[B, C, N]
        fusion_feature1 = fusion_feature1.permute(0, 2, 1) + F_point #[B, N_p, C]

        #fusion with img3d
        F_img3d = F_img3d.contiguous().view(B, C, -1)
        F_img3d = F_img3d.permute(0, 2, 1)  #[B, N_i, C]
        img_3d = self.img_mlp(F_img3d)
        img_3d = self.bn1(img_3d.permute(0, 2, 1))   #[B, C1, N_i]

        F_fusion = self.point_mlp(fusion_feature1) #[B, N_p, C1]
        F_fusion = self.bn1(F_fusion.permute(0, 2, 1)) #[B, C1, N_p]

        F_fusion_T = F_fusion.permute(0, 2, 1)  #[B, N_p, C1]
        atten_2 = torch.bmm(F_fusion_T, img_3d) * scale #[B, N_p, N_i]
        atten_2 = self.softmax(atten_2)
        atten_out_2 = torch.bmm(atten_2, img_3d.permute(0,2,1))  #[B, N_p, C1]
        fusion_feature2 = torch.cat((atten_out_2, fusion_feature1), dim=2)  #[B, N_p, C1 + C]
        fusion_feature2 = self.fusion_mlp(fusion_feature2)
        fusion_feature2 = self.bn2(fusion_feature2.permute(0, 2, 1)) #[B, C, N_p]
        fusion_feature2 = fusion_feature2.permute(0,2,1) + fusion_feature1  #[B, N_p, C]
        fusion_feature2 = self.leaky_relu(fusion_feature2)
        fusion_feature2 = fusion_feature2.permute(0, 2, 1) #[B, C, N_p]

        return fusion_feature2

if __name__ == '__main__':
    point = torch.randn(16, 512, 1024)
    img_2d = torch.randn(16, 256, 56, 56)
    img_3d = torch.randn(16, 256, 56, 56)

    model = fusion_model(256, 512, 256)
    out = model(point, img_2d, img_3d)
    print(out.size())  #[B, N, C] --> [16, 1024, 512]