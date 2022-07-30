
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
from torchvision import transforms
import pdb

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

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

class PI_dataset(Dataset):
    def __init__(self, point_path, img_path):
        super().__init__()

        self.p_path = point_path  #txt: all file path
        self.i_path = img_path

        self.point_files = self.read_file(self.p_path)
        self.img_files = self.read_file(self.i_path)

        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']


    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, index):
        point_path = self.point_files[index]
        img_path = self.img_files[index]

        Img = Image.open(img_path).convert('RGB')
        Img = Img.resize((224,224))
        Img = img_normalize(Img)

        Point, affordance_label = self.extract_point_file(point_path)

        Point,_,_ = pc_normalize(Point)
        Point = Point.transpose()
        affordance_label = self.get_affordance_label(img_path, affordance_label)

        return Img, Point, affordance_label, point_path, img_path

    def read_file(self, path):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                file_list.append(file)

            f.close()
        return file_list
    
    def extract_point_file(self, path):
        with open(path,'r') as f:
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
        affordance_label = data_array[: , 3:]

        return points_coordinates, affordance_label

    def get_affordance_label(self, str, label):
        cut_str = str.split('_')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)
        if(index == 0):
            label[:, index+1:] = 0
        elif(index == 17):
            label[:, :index] = 0
        else:
            label[:, 0:index] = 0
            label[:, index+1:] = 0
        
        return label


if __name__=='__main__':
    img_train_path = 'Data/Img_Train.txt'
    point_train_path = 'Data/Point_Train.txt'
    train_dataset = PI_dataset(point_train_path, img_train_path)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=8, shuffle=True)
    print(f'len_dataset:{len(train_dataset)}')
    print(f'len_dataloader:{len(train_loader)}')
    # for i,(img, point, label,_,img_path) in enumerate(train_loader):
    #     '''
    #     img : [B, C, H, W]
    #     point: [B, 3, N]
    #     label: [B, N, 18]
    #     '''
    #     if(img.shape[1] == 1):
    #         print(img_path)
  







