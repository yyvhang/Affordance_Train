import os
import pdb

'''
Img data rename
'''
# data_file = 'Data/Img/Train'
# object_list = os.listdir(data_file)
# object_folder = [os.path.join(data_file,x) for x in object_list]
# number = 1
# for Index,folder in enumerate(object_folder):
#     affordance_list = os.listdir(folder)
#     affordance_folder = [os.path.join(folder,y) for y in affordance_list]
#     for index,a_folder in enumerate(affordance_folder):
#         file_list = os.listdir(a_folder)
#         for i,file in enumerate(file_list):
#             old_file = os.path.join(a_folder, file)
#             new_file = a_folder + '/' + f'Image_Train_{object_list[Index]}_{affordance_list[index]}_{number}.jpg'
#             number += 1
#             os.rename(old_file,new_file)
#             print(f'{new_file} rename finish!')


'''
point data rename
'''
# point_data = 'Data/Point/Val'
# object_list = os.listdir(point_data)
# object_folder = [os.path.join(point_data,x) for x in object_list]
# number = 1
# for Index,object_file in enumerate(object_folder):
#     point_file = os.listdir(object_file)
#     for file in point_file:
#         old_name = os.path.join(object_file,file)
#         new_name = object_file + '/' + f'Point_Val_{object_list[Index]}_{number}.txt'
#         os.rename(old_name,new_name)
#         number += 1
#         print(f'{new_name} rename finish!')

'''
all train img_path in one txt
'''
# data_file = 'Data/Point/Train'
# with open('Data/Point_Train.txt','w') as f:
#     object_list = os.listdir(data_file)
#     object_folder = [data_file + '/' + x for x in object_list]

#     for object_path in object_folder:
#         affordance_list = os.listdir(object_path)
#         affordance_folder = [object_path + '/' + affordance_name for affordance_name in affordance_list]

#         for affordance_path in affordance_folder:
#             file_list = os.listdir(affordance_path)
            
#             for file in file_list:
#                 file_name = affordance_path + '/' + file
#                 f.write(f'{file_name}\n')
#                 print(f'{file_name} write finish!')

#     f.close()

'''
all train point_path in one txt
'''
data_file = 'Data/Point/Val'
with open('Data/Point_Val.txt','w') as f:
    object_list = os.listdir(data_file)
    object_folder = [data_file + '/' + x for x in object_list]

    for object_path in object_folder:
        point_files = os.listdir(object_path)
        for file in point_files:
            file_name = object_path + '/' + file
            f.write(f'{file_name}\n')
            print(f'{file_name} write finish!')

    f.close()


