import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from data_utils.dataset import PI_dataset
from model.Encoder import Fusion_model
from utils.eval import evaluating
import numpy as np


def eval(dataset,data_loader, model_path, affordance, use_gpu):
    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = Fusion_model(num_classes=18)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    results = torch.zeros((len(dataset), 2048, 18))
    targets = torch.zeros((len(dataset), 2048, 18))
    num = 0
    with torch.no_grad():
        model.eval()
        total_point = 0
        total_MSE = 0
        for i,(img, point, label,_,_) in enumerate(data_loader):
            print(f'iteration: {i} start----')
            point = point.float()
            if(use_gpu):
                img = img.to(device)
                point = point.to(device)
                label = label.to(device)

            pred = model(img, point)

            mse, point_nums = evaluating(pred, label)
            total_point += point_nums
            total_MSE += mse
            pred_num = pred.shape[0]
            # pred = pred.squeeze()
            # label = label.squeeze()
            results[num : num+pred_num, :, :] = pred
            targets[num : num+pred_num, :, :] = label
            num += pred_num

        each_mean_mse = total_MSE / total_point
        mean_mse = torch.sum(each_mean_mse)
        results = results.detach().numpy()
        targets = targets.detach().numpy()
        AP = np.zeros((targets.shape[0], targets.shape[2]))
        F1 = np.zeros((targets.shape[0], targets.shape[2]))
        AUC = np.zeros((targets.shape[0], targets.shape[2]))
        IOU = np.zeros((targets.shape[0], targets.shape[2]))
        targets = targets >= 0.5
        targets = targets.astype(int)
        IOU_thres = np.linspace(0, 1, 20)
        for i in range(AP.shape[0]):
            t = targets[i, :, :]
            p = results[i, :, :]
            for j in range(t.shape[1]):
                t_true = t[:, j]
                p_score = p[:, j]
                if np.sum(t_true) == 0:
                    F1[i, j] = np.nan
                    AP[i, j] = np.nan
                    AUC[i, j] = np.nan
                    IOU[i, j] = np.nan
                else:
                    ap = average_precision_score(t_true, p_score)
                    AP[i, j] = ap
                    p_mask = (p_score > 0.5).astype(int)
                    f1 = f1_score(t_true, p_mask)
                    F1[i, j] = f1
                    auc = roc_auc_score(t_true, p_score)
                    AUC[i, j] = auc
                    temp_iou = []
                    for thre in IOU_thres:
                        p_mask = (p_score >= thre).astype(int)
                        intersect = np.sum(p_mask & t_true)
                        union = np.sum(p_mask | t_true)
                        temp_iou.append(1.*intersect/union)
                    temp_iou = np.array(temp_iou)
                    aiou = np.mean(temp_iou)
                    IOU[i, j] = aiou
        AP = np.nanmean(AP, axis=0)
        F1 = np.nanmean(F1, axis=0)
        AUC = np.nanmean(AUC, axis=0)
        IOU = np.nanmean(IOU, axis=0)

        for i in range(AP.size):
            outstr = affordance[i] + ': ' + 'AP = ' + str(AP[i]) + ' | ' + 'AUC = ' + str(AUC[i]) + ' | ' +'aIOU = ' + str(IOU[i]) + ' | '+'mse = ' + str(each_mean_mse[i].item())
            print(outstr)

        print(f'val_mean_mse:{mean_mse.item()} | val_mean_AP:{np.nanmean(AP)} | val_mean_AUC:{np.nanmean(AUC)} | val_mean_IOU:{np.nanmean(IOU)}')

if __name__=='__main__':
    point_path = 'Data/Point_Val.txt'
    img_path = 'Data/Img_Val.txt'
    model_path = 'runs/train/train_1/best.pt'
    val_dataset = PI_dataset(point_path, img_path)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=True)
    affordance_list = ['grasp', 'contain', 'lift', 'open', 
                'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                'push', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']
    use_gpu = False
    eval(val_dataset, val_loader, model_path, affordance_list, use_gpu)