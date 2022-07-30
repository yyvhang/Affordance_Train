import numpy as np
import torch
import pdb
def evaluating(pred, label):
    #[B, N, 18]
    #MSE
    l2_distance = torch.sum(torch.pow(pred-label, 2), dim=(0,1))
    points_num = pred.shape[0] * pred.shape[1]

    return l2_distance, points_num

# pred = torch.randn(1, 2048, 18)
# label = torch.randn(1, 2048, 18)
# total = 0
# mse, points_num = evaluating(pred, label)
# total += mse
# mean_mse = mse / points_num
# total_mse = torch.sum(mse) / points_num
# best = 10000
# a = 9999999.
# if(total_mse < best):
#     print('can')
#     print(a)
# total_mse2 = torch.sum(mean_mse)
# print(mse)
# print(points_num)
# print(mean_mse)
# print(total_mse)
# print(total_mse2)