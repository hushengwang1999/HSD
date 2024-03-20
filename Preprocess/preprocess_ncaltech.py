import torch
import torchvision
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import numpy as np
from os.path import isfile, join

class NCaltech101(Dataset):
    def __init__(self, data_path='D:/下载/NCaltech101',
                 data_type='train', transform=False):

        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.clslist):
            # print (i, cls)
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if data_type == 'train':
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        if data_type != 'train':
            counts = np.unique(np.array(self.targets), return_counts=True)[1]
            class_weights = counts.sum() / (counts * len(counts))
            self.class_weights = torch.Tensor(class_weights)
        self.classes = range(101)
        self.transform = transform
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))
        self.translate=transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2), fill=0)
        self.crop=transforms.RandomCrop(48, padding=4)
        self.horizonalFilp=transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        data = self.resize(data)
        train_transform = transforms.Compose([

            transforms.RandomCrop(48, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])
        if self.transform:


            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))


        return data, label

    def __len__(self):
        return self.data_num

