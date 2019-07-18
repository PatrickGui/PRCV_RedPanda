import json
import os

from torch.utils.data import *
from PIL import Image
from torchvision.transforms import transforms

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataSet(Dataset):
    def __init__(self, root, loader=default_loader, extensions=None, transform=None, target_transform=None, root_pre=None):
        '''

        :param root: 存放图片路径和标签的json文件路径
        :param loader:
        :param extensions:
        :param transform:
        :param target_transform:
        :param root_pre: 存放图片的文件夹根目录路径
        '''
        samples = []
        for file in os.listdir(root):
            # if not index==0: #划分验证集
            if not os.path.isdir(file):
                for line in open(root+file, 'r'):
                    str_line = line.split()
                    samples.append(tuple([str_line[0], int(str_line[1])]))



        # imgs = json.load(open(root, 'r'))
        #
        #
        # image_path = [element['image_id'] for element in imgs]
        # image_label = [element['disease_class'] for element in imgs]
        # samples = list(zip(image_path, image_label))

        self.samples = samples  #2250
        self.targets = [int(s[1]) for s in samples]
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions
        self.root_pre = root_pre

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(self.root_pre+path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    ANNOTATION_TRAIN= '/media/phgui/2D6E11B1D89550E7/IDADP-PRCV2019-training/protocol/Train/'
    IMAGE_TRAIN_PRE = '/media/phgui/2D6E11B1D89550E7/IDADP-PRCV2019-training/'
    ANNOTATION_VAL = '/media/phgui/2D6E11B1D89550E7/IDADP-PRCV2019-training/protocol/Val/'
    MyDataSet(root=ANNOTATION_TRAIN, root_pre=IMAGE_TRAIN_PRE)