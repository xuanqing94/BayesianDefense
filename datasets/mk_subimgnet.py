import shutil
import os

def read_f():
    classes = []
    for l in open('./dirname_to_label.txt', 'r'):
        name, id = l.split(' ')
        id = int(id)
        if id >= 151 and id < 294:
            classes.append(name)
    return classes

if __name__ == "__main__":
    source_dir = '/mnt/lustre/share/images/train'
    target_dir = '/mnt/lustre/wuchongruo/projects/xq/data/train'
    classes = read_f()
    for i, name in enumerate(classes):
        print(f'[{i}/{len(classes)}]{name}')
        
        src = source_dir + '/' + name
        dest = target_dir + '/' + name

        #shutil.copytree(src, dest)
        cmd = '''cp -r ''' + src + " " + dest
        os.system(cmd)
        
    print('Done')




