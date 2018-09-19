import shutil, os
import glob
from shutil import copyfile
import pdb

def read_f():
    id_classes = {}
    for l in open('./dirname_to_label.txt', 'r'):
        name, id = l.split(' ')
        id = int(id)
        id_classes[id] = name
    return id_classes

if __name__ == "__main__":
    #source_dir = '/mnt/lustre/wuchongruo/projects/xq/data/prepare/val'
    source_dir = '/mnt/lustre/share/images/val'
    target_dir = '/mnt/lustre/wuchongruo/projects/xq/BayesianDefense/data/sngan_dog_cat_val'
    id_classes = read_f()
    
    #all_files = glob.glob(source_dir+'/*.JPEG')

    all_files_id = {}
    with open('/mnt/lustre/share/images/meta/val.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            file_name, class_id = line.split(' ') 
            class_id = int(class_id)
            all_files_id[file_name] = class_id
    
    for ind, cur_file_name in enumerate(list(all_files_id.keys())):
        cur_file_id = all_files_id[ cur_file_name ]

        if cur_file_id >= 151 and cur_file_id < 294:
            cur_file_class_name = id_classes[cur_file_id]
            class_dir_path = os.path.join(target_dir, cur_file_class_name)
            if not os.path.exists(class_dir_path):
                os.mkdir(class_dir_path)

            source_file_path = os.path.join(source_dir, cur_file_name)
            dest_file_path = os.path.join(target_dir, cur_file_class_name, cur_file_name)
            copyfile(source_file_path, dest_file_path)


    '''
    for ind, each_file_full in enumerate(all_files):
        print(ind)
        #print(each_file_full)
        each_file_name = each_file_full.split('/')[-1]
        each_file_class = 'n' + each_file_name.split('.')[0].split('_')[-1]
        
        #print(each_file_class)
        each_file_id = classes_id[each_file_class]
        if each_file_id >= 151 and each_file_id < 294:
            class_dir_path = os.path.join(target_dir, each_file_class)
            if not os.path.exists(class_dir_path):
                os.mkdir(class_dir_path)
            
            dest_file_path = os.path.join(target_dir, each_file_class, each_file_name)
            copyfile(each_file_full, dest_file_path)
    '''

















