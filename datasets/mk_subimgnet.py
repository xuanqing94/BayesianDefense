import shutil

def read_f():
    classes = []
    for l in open('./dirname_to_label', 'r'):
        name, id = l.split(' ')
        id = int(id)
        if id >= 151 and id < 294:
            classes.append(name)
    return classes

if __name__ == "__main__":
    source_dir = './path/to/imagenet'
    target_dir = './path/to/subimagenet'
    classes = read_f()
    for i, name in enumerate(classes):
        print(f'[{i}/{len(classes)}]{name}')
        shutil.copy(source_dir + '/name', target_dir)
    print('Done')
