import argparse
from ast import parse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils.dataset import PI_dataset
from model.Baseline_v1 import model_encoder
from utils.loss import Final_Loss
from utils.eval import evaluating
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import numpy as np
import os
import pdb
import logging
#import visdom

def main(opt):
    if opt.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    save_path = opt.save_dir + opt.name
    foler = os.path.exists(save_path)
    if not foler:
        os.makedirs(save_path)

    loger = logging.getLogger('Training')
    loger.setLevel(logging.INFO)
    logging.basicConfig(filename='log/2022_7_24.log', level=logging.INFO)
    def log_string(str):
        loger.info(str)
        print(str)

    #vis = visdom.Visdom(env='model_Training')
    #import data
    img_train_path = opt.img_train_path
    point_train_path = opt.point_train_path
    img_val_path = opt.img_val_path
    point_val_path = opt.point_val_path

    log_string('Start loading train data---')
    train_dataset = PI_dataset(point_train_path, img_train_path)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=8 ,shuffle=True)
    log_string(f'train data loading finish, loading data files:{len(train_dataset)}')

    log_string('Start loading val data---')
    val_dataset = PI_dataset(point_val_path, img_val_path)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True)
    log_string(f'val data loading finish, loading data files:{len(val_dataset)}')

    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    # model, optimizer, loss
    img_pre_path = 'runs/pre_train/resnet50-19c8e357.pth'
    point_pre_path = 'runs/pre_train/pointnet2.pth'
    model = model_encoder(img_pre_path, point_pre_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.decay_rate)
    criterion = Final_Loss()

    if opt.resume:
        model_checkpoint = torch.load(opt.checkpoint_path, map_location='cuda:1')
        model.load_state_dict(model_checkpoint['model'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['Epoch']
    else:
        #model.apply(weight_init)
        start_epoch = -1

    # def bn_momentum_adjust(m, momentum):
    #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         m.momentum = momentum

    # MOMENTUM_ORIGINAL = 0.1
    # MOMENTUM_DECCAY = 0.5
    # MOMENTUM_DECCAY_STEP = opt.step_size

    model = model.to(device)
    criterion = criterion.to(device)

    best_map = 0.0
    '''
    Training
    '''
    for epoch in range(start_epoch+1, opt.Epoch):
        log_string(f'Epoch:{epoch} strat-------')

        # momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        # if momentum < 0.01:
        #     momentum = 0.01
        # print('BN momentum updated to: %f' % momentum)
        # model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_loader)
        loss_sum = 0
        total_MSE = 0
        total_point = 0
        model = model.train()
        log_string(f'cuda memorry:{torch.cuda.memory_allocated(device=opt.gpu) / (1024*1024)}')
        for i,(img, point, label,_,_) in enumerate(train_loader):
            optimizer.zero_grad()
            point = point.float()
            if(opt.use_gpu):
                img = img.to(device)
                point = point.to(device)
                label = label.to(device)

            
            pred = model(img, point)
            loss = criterion(pred, label)
            mse, point_nums = evaluating(pred, label)
            mse = torch.sum(mse)
            log_string(f'Epoch:{epoch} | iteration:{i} | loss:{loss} | mse:{mse}')
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            total_point += point_nums
            total_MSE += mse.item()

        log_string(f'cuda memorry:{torch.cuda.memory_allocated(device=opt.gpu) / (1024*1024)}')
        mean_loss = loss_sum / num_batches
        mean_mse = total_MSE / total_point
        log_string(f'Epoch:{epoch} | mean_loss:{mean_loss} | mean_mse:{mean_mse}')
        # if(epoch == 0):
        #     cpu_mean_loss = cpu_mean_loss.cpu()
        #     cpu_mean_loss = mean_loss.detach().numpy()
        #     window = vis.line(
        #         X = np.array([epoch]),
        #         Y = np.array([cpu_mean_loss]),
        #         opts={
        #             'titile' : "Loss",
        #             'showlegend' : True,
        #             'xlabel' : "Epoch",
        #             'ylabel' : "loss",
        #         },
        #     )
        # else:
        #     cpu_mean_loss = cpu_mean_loss.cpu()
        #     cpu_mean_loss = mean_loss.detach().numpy()
        #     vis.line(
        #         X = np.array([epoch]),
        #         Y = np.array([cpu_mean_loss]),
        #         win = window,
        #         update = 'append'
        #     )


        if((epoch+1) % 5==0):
            model_path = save_path + '/Epoch_' + str(epoch+1) + '.pt'
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'Epoch': epoch
            }
            torch.save(checkpoint, model_path)
            log_string(f'model saved at {model_path}')
        
        results = torch.zeros((len(val_dataset), 2048, 18))
        targets = torch.zeros((len(val_dataset), 2048, 18))
        '''
        Evalization
        '''
        num = 0
        with torch.no_grad():
            log_string(f'EVALUATION strat-------')
            num_batches = len(val_loader)
            # loss_sum = 0
            total_MSE = 0
            total_point = 0
            model = model.eval()
            for i,(img, point, label,_,_) in enumerate(val_loader):
                print(f'iteration: {i} start----')
                point = point.float()
                if(opt.use_gpu):
                    img = img.to(device)
                    point = point.to(device)
                    label = label.to(device)
                
                pred = model(img, point)

                mse, point_nums = evaluating(pred, label)
                mse = torch.sum(mse)
                total_point += point_nums
                total_MSE += mse.item()
                pred_num = pred.shape[0]
                print(f'num:{num}, pred_num:{pred_num}')
                results[num : num+pred_num, :, :] = pred
                targets[num : num+pred_num, :, :] = label
                num += pred_num

            # mean_loss = loss_sum / num_batches
            mean_mse = total_MSE / total_point
            results = results.detach().numpy()
            targets = targets.detach().numpy()
            log_string(f'cuda memorry:{torch.cuda.memory_allocated(device=opt.gpu)/ (1024*1024)}')
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
            log_string(f'val_mean_mse:{mean_mse} | val_mean_AP:{np.nanmean(AP)} | val_mean_AUC:{np.nanmean(AUC)} | val_mean_IOU:{np.nanmean(IOU)}')

            current_map = np.nanmean(AP)
            if(current_map > best_map):
                best_map = current_map
                best_model_path = save_path + '/best.pt'
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'Epoch': epoch
                }
                torch.save(checkpoint, best_model_path)
                log_string(f'best model saved at {best_model_path}')





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_train_path', type=str, default='Data/Img_Train.txt', help='all train_img path in a txt file')
    parser.add_argument('--point_train_path', type=str, default='Data/Point_Train.txt', help='all train_point path in a txt file')
    parser.add_argument('--img_val_path', type=str, default='Data/Img_Val.txt', help='all val_img path in a txt file')
    parser.add_argument('--point_val_path', type=str, default='Data/Point_Val.txt', help='all val_point path in a txt file')
    parser.add_argument('--batch_size', type=int, default=16, help='train batch size')
    parser.add_argument('--lr_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu device id')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--Epoch', type=int, default=200, help='total epoch')
    parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
    parser.add_argument('--save_dir', type=str, default='runs/train/', help='path to save .pt model while training')
    parser.add_argument('--name', type=str, default='train_4', help='training name to classify each training process')
    parser.add_argument('--resume', type=str, default=False, help='start training from previous epoch')
    parser.add_argument('--checkpoint_path', type=str, default='runs/train/train_3/Epoch_15.pt', help='checkpoint path')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for momentum decay [default: every 10 epochs]')

    opt = parser.parse_args()

    main(opt)