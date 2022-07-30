from cv2 import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class Final_Loss(nn.Module):
    def __init__(self):
        super(Final_Loss, self).__init__()
        self.gamma = 0
        self.alpha = 0

    def forward(self, pred, target):
        #[B, N, 18]
        temp1 = -torch.mul(pred**self.gamma,
                           torch.mul(1-target, torch.log(1-pred+1e-6)))
        temp2 = -torch.mul((1-pred)**self.gamma,
                           torch.mul(target, torch.log(pred+1e-6)))
        temp = temp1+temp2
        CELoss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        DICELoss = torch.sum(temp3)

        return CELoss+1.0*DICELoss

# loss = Final_Loss()
# pred = torch.randn(16, 2048, 18)
# target = torch.randn(16, 2048, 18)
# sigmod = nn.Sigmoid()
# pred = sigmod(pred)
# target = sigmod(target)
# result = loss(pred, target)
# x = torch.randn(2,3,8)   #[B, C, N]  must be this shape
# target = torch.randn(2,3,8)
# loss = nn.CrossEntropyLoss()
# loss_value = loss(x, target)
# #pdb.set_trace()
# x = -F.log_softmax(x, dim=1)
# loss2 = target * x
# loss2 = torch.sum(loss2, dim=1)
# print(f'loss1:{loss_value}')
# print(f'loss2:{loss2}')


