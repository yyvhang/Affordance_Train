import pickle as pkl
import os
import pdb
from turtle import width
from matplotlib.pyplot import axis
import numpy as np
import open3d as o3d


Affordance_label_list = ['grasp', 'contain', 'lift', 'openable', 
                        'layable', 'sittable', 'support', 'wrap_grasp', 'pourable', 'move', 'displaY',
                        'pushable', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']
color_list = [[252, 19, 19], [249, 113, 45], [247, 183, 55], [251, 251, 11], [178, 244, 44], [255, 0, 0], 
              [0, 0, 255], [25, 248, 99], [46, 253, 184], [40, 253, 253], [27, 178, 253], [28, 100, 243], 
              [46, 46, 125], [105, 33, 247], [172, 10, 253], [249, 47, 249], [253, 51, 186], [250, 18, 95]]
color_list = np.array(color_list)
'''
use txt dataformat
'''
txt_path = 'D:/USTC/Research/Code_project/full-shape/Data/Point/Train/StorageFurniture/points_train_8207.txt'
def visual_txt(Txt_path):
    with open(Txt_path,'r') as f:
        coordinates = []
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            # coordinate = [float(x) for x in data[2:5]]
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)  #包含label信息的np数组
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]
        visual_point = o3d.geometry.PointCloud() #创建可视化对象
        visual_point.points = o3d.utility.Vector3dVector(points_coordinates)
        # R = visual_point.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4)) #获取旋转矩阵
        # visual_point.rotate(R) #旋转
        # color = np.random.random((2048, 3))
        color = np.zeros((2048,3))
        for i,point_affordance in enumerate(affordance_label):
            if(np.max(point_affordance) > 0):
                color_index = np.argmax(point_affordance)
                color[i] = color_list[color_index]

        visual_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

        o3d.visualization.draw_geometries([visual_point])
        f.close()

'''
visualize pred and GT
'''
def visual_pred(affordance_pred, GT_path):
    with open(GT_path,'r') as f:
        coordinates = []
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            # coordinate = [float(x) for x in data[2:5]]
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)  #包含label信息的np数组
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]

        gt_point = o3d.geometry.PointCloud() #创建可视化对象
        gt_point.points = o3d.utility.Vector3dVector(points_coordinates)

        pred_point = o3d.geometry.PointCloud()
        pred_point.points = o3d.utility.Vector3dVector(points_coordinates)
        # R = visual_point.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4)) #获取旋转矩阵
        # visual_point.rotate(R) #旋转
        # color = np.random.random((2048, 3))
        color = np.zeros((2048,3))
        reference_color = np.array([204, 255, 102])
        # for i,point_affordance in enumerate(affordance_label):
        #     if(np.max(point_affordance) > 0):
        #         color_index = np.argmax(point_affordance)
        #         color[i] = color_list[color_index]
        for i, point_affordacne in enumerate(affordance_label):
            if(np.max(point_affordacne) > 0.1):
                scale_i = np.max(point_affordacne)
                color[i] = reference_color * scale_i
        gt_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

        pred_color = np.zeros((2048,3))
        # for i,pred in enumerate(affordance_pred):
        #     if(np.max(pred) > 0.2):
        #         color_index = np.argmax(pred)
        #         pred_color[i] = color_list[color_index]
        for i, pred in enumerate(affordance_pred):
            if(np.max(pred) > 0.1):
                scale_i = np.max(pred)
                pred_color[i] = reference_color * scale_i
        pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)

        o3d.visualization.draw_geometries([gt_point], window_name='GT point', width=1000, height=800)
        o3d.visualization.draw_geometries([pred_point], window_name='pred point', width=1000, height=800)
        f.close()

'''
 use pkl dataformat
'''
pkl_path = 'D:/USTC/Research/Code_project/full-shape/full_shape_val_data.pkl'
def visual_pkl(pkl_path):
    points_file = open(pkl_path, 'rb')
    temp_data = pkl.load(points_file)
    for index, info in enumerate(temp_data):
        shape_id = info['shape_id'] #str
        affordance_label_list = info['affordance']
        object_class = info['semantic class']  #str
        points_coordinate = info['full_shape']['coordinate']  #[2048, 3]
        affordance_label = info['full_shape']['label']  #
        Points_data = points_coordinate
        pdb.set_trace()
        if(index == 2000):
            # R = Points_data.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4))
            # Points_data.rotate(R,center=(0, 0, 0))
            print(object_class)
            visual_point = o3d.geometry.PointCloud() #创建可视化对象
            visual_point.points = o3d.utility.Vector3dVector(Points_data)
            R = visual_point.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4)) #获取旋转矩阵
            visual_point.rotate(R) #旋转
            color = np.random.random((2048, 3))
            visual_point.colors = o3d.utility.Vector3dVector(color)
            o3d.visualization.draw_geometries([visual_point])

        for aff in affordance_label_list:
            temp = affordance_label[aff].astype(np.float32).reshape(-1, 1)
            pdb.set_trace()
            Points_data = np.concatenate((Points_data, temp), axis=1)
