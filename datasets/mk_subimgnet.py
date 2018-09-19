import shutil

def read_f():
    classes = []
    for l in open('./dirname_to_label.txt', 'r'):
        name, id = l.split(' ')
        id = int(id)
        if id >= 151 and id < 294:
            classes.append(name)
    return classes

if __name__ == "__main__":
    source_dir = '/mnt/lustre/wuchongruo/projects/xq/data/prepare/train'
    target_dir = '/mnt/lustre/wuchongruo/projects/xq/BayesianDefense/data/train'
    classes = read_f()
    for i, name in enumerate(classes):
        print(f'[{i}/{len(classes)}]{name}')
        shutil.copytree(source_dir + '/' + name, target_dir + '/' + name)
    print('Done')




