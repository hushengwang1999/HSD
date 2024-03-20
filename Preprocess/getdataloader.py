from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from Preprocess.augment import Cutout, CIFAR10Policy, dvs_aug
import random
import numpy as np

# your own data dir
DIR = {'CIFAR10': 'E:\datasets', 'CIFAR100': 'E:\datasets', 'ImageNet': 'YOUR_IMAGENET_DIR'}


def dvs_aug(data):
    flip = random.random() > 0.5
    if flip:
        data = np.flip(data, axis=3)
    off1 = random.randint(-5, 5)
    off2 = random.randint(-5, 5)
    data = np.roll(data, shift=(off1, off2), axis=(2, 3))
    return data


def apply_random_affine(data):
    # Create a RandomAffine transform with the desired parameters
    random_affine = transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2), fill=0)

    # Apply the transform to the data
    transformed_data = random_affine(torch.from_numpy(data))

    # Return the transformed data
    return transformed_data


##自定义数据增强
def shuffle(data):
    data_shuffled = data[np.random.permutation(data.shape[0]), :, :, :]
    return data_shuffled


# def GetCifar10(batchsize, attack=False):
#     if attack:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   Cutout(n_holes=1, length=16)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor()])
#     else:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                                   Cutout(n_holes=1, length=8)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#     train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
#     test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
#     train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
#     test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
#     return train_dataloader, test_dataloader

def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader


def GetCifar100(batchsize):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                       std=[n / 255. for n in [68.2, 65.4, 70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],
                                                     std=[n / 255. for n in [68.2, 65.4, 70.4]])])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader


def GetImageNet(batchsize):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])

    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8, sampler=train_sampler,
                                  pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler)
    return train_dataloader, test_dataloader


# def collate_fn(batch, split_ratio=0.5):
#     # 将batch拆分成数据和标签
#     data = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#
#     # 计算分割点的索引
#     split_idx = int(len(data[0]) * split_ratio)
#
#     # 构造第一个数据集
#     data_1 = [d[:split_idx, :] for d in data]
#     dataset_1 = list(zip(data_1, labels))
#
#     # 构造第二个数据集
#     data_2 = [d[split_idx:, :] for d in data]
#     dataset_2 = list(zip(data_2, labels))
#
#     return dataset_1, dataset_2

def GetCifar10dvsNet(batchsize):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda import amp
    from spikingjelly.clock_driven import functional, surrogate, layer, neuron
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
    from spikingjelly.datasets.n_caltech101 import NCaltech101
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import time
    import os
    import argparse
    import numpy as np
    #data_dir = '/data/users/husw/datasets/cifar10-dvs'
    data_dir='/data/users/husw/datasets/N_caltech101'
    T = 10
    _seed_ = 1000
    b = 128  # 批次
    j = 16  # 线程数量
    origin_set=NCaltech101(root=data_dir, data_type='frame', frames_number=T, split_by='number')
    #origin_set = CIFAR10DVS(root=data_dir, data_type='frame', frames_number=T, split_by='number')
    from spikingjelly.datasets.__init__ import split_to_train_test_set
    # cache_dir = os.path.join('/data/users/husw/datasets/cifar10-dvs/cifar10dvs_cache1/',
    #                          f'seed_{_seed_}_T_{T}_number_no_aug/')
    cache_dir=os.path.join('/data/users/husw/datasets/N_caltech101/cache/',f'seed_{_seed_}_T_{T}_number_no_aug/')
    if os.path.exists(cache_dir):
        train_set = torch.load(os.path.join(cache_dir, 'train.pt'))

        test_set = torch.load(os.path.join(cache_dir, 'test.pt'))
    else:
        train_set, test_set = split_to_train_test_set(0.9, origin_set, 10)
        os.mkdir(cache_dir)
        torch.save(train_set, os.path.join(cache_dir, 'train.pt'))
        torch.save(test_set, os.path.join(cache_dir, 'test.pt'))

    # test_T=10
    # origin_set1 = CIFAR10DVS(root=data_dir, data_type='frame', frames_number=test_T, split_by='number')
    # from spikingjelly.datasets.__init__ import split_to_train_test_set
    # cache_dir = os.path.join('/data/users/husw/datasets/cifar10-dvs/cifar10dvs_cache1/',
    #                          f'seed_{_seed_}_T_{T}_number_no_aug/')
    # if os.path.exists(cache_dir):
    #     train_set1 = torch.load(os.path.join(cache_dir, 'train.pt'))
    #
    #     test_set1 = torch.load(os.path.join(cache_dir, 'test.pt'))
    # else:
    #     train_set1, test_set1 = split_to_train_test_set(0.9, origin_set1, 10)
    #     os.mkdir(cache_dir)
    #     torch.save(train_set1, os.path.join(cache_dir, 'train.pt'))
    #     torch.save(test_set1, os.path.join(cache_dir, 'test.pt'))

    # train_set1, test_set1 = collate_fn(train_set1,split_ratio=0.5)
    # train_set2, test_set2 = collate_fn(test_set1, split_ratio=0.5)

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=b,
        shuffle=True,
        num_workers=j,
        drop_last=True,
        pin_memory=True)

    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=b,
        shuffle=False,
        num_workers=j,
        drop_last=False,
        pin_memory=True)

    # train_data_loader1 = DataLoader(
    #     dataset=train_set1,
    #     batch_size=b,
    #     shuffle=True,
    #     num_workers=j,
    #     drop_last=True,
    #     pin_memory=True)
    #
    # test_data_loader1 = DataLoader(
    #     dataset=test_set1,
    #     batch_size=b,
    #     shuffle=False,
    #     num_workers=j,
    #     drop_last=False,
    #     pin_memory=True)
    # train_data_loader2 = DataLoader(
    #     dataset=train_set2,
    #     batch_size=b,
    #     shuffle=True,
    #     num_workers=j,
    #     drop_last=True,
    #     pin_memory=True)
    #
    # test_data_loader2 = DataLoader(
    #     dataset=test_set2,
    #     batch_size=b,
    #     shuffle=False,
    #     num_workers=j,
    #     drop_last=False,
    #     pin_memory=True)
    # return train_data_loader1, test_data_loader1,train_data_loader2,test_data_loader2
    return train_data_loader, test_data_loader