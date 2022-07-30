import pickle as pkl
import os
import pdb
import numpy as np

Path = '/Users/yangyuhang21583/Desktop/working/School/Code_project/full-shape/full_shape_train_data.pkl'
points_file = open(Path, 'rb')
temp_data = pkl.load(points_file)
for index, info in enumerate(temp_data):
    shape_id = info['shape_id'] #str
    affordance_label_list = info['affordance']
    object_class = info['semantic class']  #str
    points_coordinate = info['full_shape']['coordinate']  #[2048, 3]
    affordance_label = info['full_shape']['label']  #
    Points_data = points_coordinate
    for aff in affordance_label_list:
        temp = affordance_label[aff].astype(np.float32).reshape(-1, 1)
        Points_data = np.concatenate((Points_data, temp), axis=1)

    with open(f'/Users/yangyuhang21583/Desktop/working/School/Code_project/full-shape/Data/Train/points_train_{index}.txt','w') as f:
        for per_point in range(len(Points_data)):
            per_coordinate = Points_data[per_point][:3]
            str_coordinate = ''
            for i in range(3):
                str_coordinate += (str(per_coordinate[i]) + ' ')
            per_affordance = Points_data[per_point][3:]
            str_affordance = ''
            for j in range(18):
                str_affordance += (str(per_affordance[j]) + ' ')
            f.writelines(f"{shape_id} {object_class} {str_coordinate}{str_affordance}\n")
        f.close()
    #pdb.set_trace()

# txt_path = '/Users/yangyuhang21583/Desktop/working/School/Code_project/full-shape/Data/Train/a.txt'
# with open(txt_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip('\n')
#         line = line.strip(' ')
#         pdb.set_trace()
#         data = line.split(' ')

# for index in range(10):
#     with open(f"/Users/yangyuhang21583/Desktop/working/School/Code_project/full-shape/Data/Train/points_{index}.txt", 'w') as ff:
#         for num in range(5):
#             num = str(num)
#             ff.write(f'{num}\n')
#         ff.close()
