import torch
import argparse
import numpy as np
import pdb
from PIL import Image
from model.Baseline_v1 import model_encoder
from utils.visualization import visual_pred
from torchvision import transforms
import matplotlib.pyplot as plt

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def img_normalize(img):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.GaussianBlur(kernel_size=7, sigma=(1.0, 10.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(img)
    return img

def extract_point_file(point_path):
    with open(point_path,'r') as f:
        coordinates = []
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.strip(' ')
        data = line.split(' ')
        coordinate = [float(x) for x in data[2:]]
        coordinates.append(coordinate)
    data_array = np.array(coordinates)  #包含label信息的np数组
    points_coordinates = data_array[:, 0:3]

    return points_coordinates

def inference(opt):
    model = model_encoder(img_model_path='a', point_model_path='b', num_class=18)
    checkpoint = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    Epoch = checkpoint['Epoch']
    Img = Image.open(opt.img_path).convert('RGB')
    Img = Img.resize((224,224))
    plt.imshow(Img)
    plt.show()
    Img = img_normalize(Img)
    Img = torch.unsqueeze(Img, 0)

    points = extract_point_file(opt.point_path)
    points = pc_normalize(points)
    points = points.transpose(1, 0)
    points = torch.from_numpy(points)
    points = torch.unsqueeze(points, 0)
    points = points.float()

    pred = model(Img, points)
    pred = torch.squeeze(pred)
    pred = pred.detach().numpy()
    visual_pred(pred, opt.point_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='runs/train/best.pt', help='model path')
    parser.add_argument('--img_path', type=str, default='Data/Img/Val/Knife/grasp/Image_Val_Knife_grasp_890.jpg', help='test img path')
    parser.add_argument('--point_path', type=str, default='Data/Point/Val/Knife/Point_Val_Knife_890.txt', help='test point path')
    
    opt = parser.parse_args()

    inference(opt)