import torch
import torch.multiprocessing as mp
import argparse
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d, \
    replace_batchnorm_with_tdbatchnorm
from ImageNet.train import main_worker
import torch.nn as nn
import os
from torchvision import datasets, transforms
from modules import TCL, MyFloor, ScaledNeuron, StraightThrough, LIFSpike
from Models.ANN_DVS_Gesture import *
from torch.utils.data import DataLoader
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from Models.ANN_DVS_Gesture import *

torch.backends.cudnn.enabled = False
CUDA_VISIBLE_DEVICES = "cuda:0"
import torch.utils.data as data


def dvs_aug(data):
    # flip = random.random() > 0.5
    # if flip:
    #     data = np.flip(data, axis=3)
    RandomHorizontalFlip = transforms.RandomHorizontalFlip()

    # Apply the transform to the data
    transformed_data = RandomHorizontalFlip(torch.from_numpy(data))
    return transformed_data


def apply_random_affine(data):
    # Create a RandomAffine transform with the desired parameters
    random_affine = transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2), fill=0)

    # Apply the transform to the data
    transformed_data = random_affine(torch.from_numpy(data))

    # Return the transformed data
    return transformed_data


def roll(data):
    data = torch.from_numpy(data)
    off1 = random.randint(-5, 5)
    off2 = random.randint(-5, 5)
    data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

    return data;


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('action', default='train', type=str, help='Action: train or test.')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=64, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=64, type=int,
                        help='Training epochs')  # better if set to 300 for CIFAR dataset
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--l', default=10, type=int, help='L')
    parser.add_argument('--t', default=10, type=int, help='T')
    # parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--data', type=str, default='cifar10dvs')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    args = parser.parse_args()

    seed_all()
    # only ImageNet using multiprocessing,
    if args.gpus > 1:
        if args.data.lower() != 'imagenet':
            AssertionError('Only ImageNet using multiprocessing.')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    else:
        # preparing data
        data_dir = '/public/home/husw/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron-main/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron-main/datasets/DVS128Gesture'
        train_set = DVS128Gesture(data_dir, train=True, data_type='frame', split_by='number', frames_number=args.t,transform=apply_random_affine)
        test_set = DVS128Gesture(data_dir, train=False, data_type='frame', split_by='number', frames_number=args.t)

        workers = 16
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True,
                                                   num_workers=workers, drop_last=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False,
                                                  num_workers=workers, drop_last=False, pin_memory=True)

        # preparing model
        model = VGG(channels=128)
        # model = replace_maxpool2d_by_avgpool2d(model)
        model = replace_activation_by_floor(model, t=args.l)
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = None
        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
        else:
            raise NotImplementedError(args.opt)

        #对应于Pre-training
        if args.action == 'train':
            train_ann_dvsg(train_loader, test_loader, optimizer, model, args.epochs, args.device, criterion, args.lr,args.wd, args.id)

        #对应于Fine-tuning
        elif args.action == 'test' or args.action == 'evaluate':

            large_model = model
            large_model.load_state_dict(
                torch.load('./saved_models_std_dvsg_l_16/' + args.id + '.pth', map_location=args.device))
            print('large_model:', large_model)

            model.load_state_dict(
                torch.load('./saved_models_std_dvsg_l_16/' + args.id + '.pth', map_location=args.device))
            small_model = replace_activation_by_neuron1(model)

            print('small_model:', small_model)

            max_acc = 0.0
            max_activate = 0.0

            large_model.to(args.device)
            small_model.to(args.device)

            #train_snn1(large_model,small_model, args.device, train_loader, test_loader, criterion,args.epochs, lr=1e-5, wd=5e-4)
            train_snn2(large_model, small_model, args.device, train_loader,test_loader, criterion, args.epochs, lr=1e-3,wd=5e-4)
            #train_snn3(large_model, small_model, args.device,train_loader, test_loader,criterion, args.epochs, lr=1e-3, wd=5e-4)

