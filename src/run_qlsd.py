#!/usr/bin/env python
# coding: utf-8

import os
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from qlsd import Qlsd
from models.lenet5 import LeNet5
from torchvision import transforms
from distutils.util import strtobool
from torch._C import default_generator
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model
# from utils.generate_imbalanced_dataset import imbalanced_dataset

# Save the user choice of settings
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="resnet20_frn_swish", choices=["resnet20_frn_swish"],
                    help='set the model name')
parser.add_argument('-n', '--num_iter', default=2, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_name', default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100", "SVHN"],
                    help='set the dataset')
parser.add_argument('-i', '--imbalance', default='False', help='if True we generate imbalanced datasets')
parser.add_argument('--proportion', default=None, type=float, help='set the imbalanced parameter')
parser.add_argument('-g', '--ngpu', default=1, type=int, help='set the number of gpus')
parser.add_argument('-l', '--learning_rate', default=1e-05, type=float, help='set the learning rate')
parser.add_argument('--momentum', default=.9, type=float, help='set the momentum')
parser.add_argument('-c', '--compression_parameter', default=2 ** 8, type=float, help='set the compression parameter')
parser.add_argument('-u', '--use_memory_term', default='True', help='if True we use a memory term')
parser.add_argument('-v', '--svrg_epoch', default=np.infty, type=int, help='set the svrg parameter')
parser.add_argument('-p', '--num_participants', default=5, type=int, help='set the number of participating clients')
parser.add_argument('-N', '--batch_size', default=64, type=int, help='set the mini-batch size')
parser.add_argument('-t', '--thinning', default=1, type=int, help='set the thinning')
parser.add_argument('-b', '--t_burn_in', default=0, type=int, help='set the burn in period')
parser.add_argument('-w', '--weight_decay', default=5, type=float, help='set the parameter of the gaussian prior')
parser.add_argument('--save_samples', default='True', help='if True we save the samples')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
args = parser.parse_args()

# Print the local torch version
print(f"Torch Version {torch.__version__}")

print("os.path.abspath(__file__) =", os.path.abspath(__file__))
path = os.path.abspath('..')

# Save the path to store the data
path_workdir = './'
path_figures = path_workdir + '/figures'
path_variables = path_workdir + '/variables'
path_csv = './'
path_dataset = path_workdir + '/dataset'
path_save_samples = path_variables + '/samples-qlsd'

print(path_save_samples)

# Create the directory if it does not exist
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_variables, exist_ok=True)
os.makedirs(path_dataset, exist_ok=True)
if strtobool(args.save_samples):
    os.makedirs(path_save_samples, exist_ok=True)

# Number of worker to distribute the data
num_workers = 20

# Set random seed for reproducibility
seed_np = args.seed if args.seed != -1 else None
seed_torch = args.seed if args.seed != -1 else default_generator.seed()
np.random.seed(seed_np)
torch.manual_seed(seed_torch)
torch.cuda.manual_seed(seed_torch)

# Start the timer
startTime = time.time()

# Load the function associated with the chosen dataset
dataset = getattr(torchvision.datasets, args.dataset_name)

# Define the transformation
normalize = [0.1307, 0.3081] if args.dataset_name == 'MNIST' else [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*normalize),
])

# Define the parameter of the dataset
params_train = {"root": path_dataset, "train": True, "transform": transform}  # , "download": True
params_test = {"root": path_dataset, "train": False, "transform": transform}

# Define the datasets
trainset = dataset(**params_train)
split_size = len(trainset) // num_workers + (1 if len(trainset) % num_workers != 0 else 0)
trainloader = DataLoader(trainset, split_size, shuffle=False)
if strtobool(args.imbalance):
    print("\n--- Generate imbalanced datasets ---\n")
    trainloader = imbalanced_dataset(trainloader, num_workers, args.proportion)
# else:
#     trainloader = imbalanced_dataset(trainloader, num_workers, proportion=1 / num_workers)

# Define the testloader
batch_size_test = 500
testset = dataset(**params_test)
testloader = DataLoader(testset, batch_size_test, shuffle=False)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# Load the saved networks
if args.dataset_name == 'MNIST':
    net = LeNet5()
else:
    pretrained = True
    net = get_model('resnet20_cifar10', pretrained=pretrained)

# We start from the last sample
epoch_init = -1
t_burn_in = args.t_burn_in
last_checkpoint = True
if last_checkpoint:
    fmax = -1
    for f in os.listdir(path_save_samples):
        if int(f) > fmax:
            fmax = int(f)
    if fmax > -1:
        epoch_init = fmax
        t_burn_in = args.thinning
        print(f'--- Load the last checkpoint: {fmax} ---\n ')
        net.load_state_dict(torch.load(os.path.join(path_save_samples, str(fmax))))
net.to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    net = nn.DataParallel(net, list(range(args.ngpu)))


# To count the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


print('Number of parameters for one neural network is: %s.' % count_parameters(net))

# Mini-batch size of each worker
mini_batch_sizes = args.batch_size * np.ones(num_workers, dtype=int)

# Define the QLSD sampler
model = Qlsd(net, num_workers, args.svrg_epoch, args.compression_parameter, strtobool(args.use_memory_term))

# Performs num_iter iterations of QLSD
model_state_dict, save_stats = model.run(trainloader, testloader, args.num_iter, args.weight_decay / num_workers,
                                         args.learning_rate, mini_batch_sizes, args.num_participants, t_burn_in,
                                         args.thinning, epoch_init, strtobool(args.save_samples), path_save_samples)
