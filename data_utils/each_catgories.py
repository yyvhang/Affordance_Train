from copy import copy
import os
import pdb
import shutil
txt_path = 'D:/USTC/Research/Code_project/full-shape/Data/Val/'
save_dir = 'D:/USTC/Research/Code_project/full-shape/Data/Val_class'
txt_path_list = os.listdir(txt_path)
for path in txt_path_list:
    points_path = os.path.join(txt_path,path)
    with open(points_path, 'r') as f:
        first_line = f.readlines()[0]
        first_line = first_line.strip(' ')
        first_line = first_line.split(' ')
        file_name = save_dir + '/' + first_line[1] + '/'+ path
        if(os.path.exists(file_name)):
            continue
        shutil.copyfile(points_path,file_name)
        print(f'file:{path}--done!')
        f.close()
