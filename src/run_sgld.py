#!/usr/bin/env python
# coding: utf-8

import os
import copy
import time
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from sgld_new import Sgld
from torchvision import transforms
from distutils.util import strtobool
from torch._C import default_generator
from torch.utils.data import DataLoader
from utils import load_dataset
from utils.sgld_tools import predictions
from utils.metrics import agreement, total_variation_distance
from utils.uncertainties_tools import PostNet, confidence, ECE, BS, Predictive_entropy, AUC, accuracy_confidence, calibration_curve

# Save the user choice of settings 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="resnet20_frn_swish", choices=["resnet20_frn_swish"],
                    help='set the model name')
parser.add_argument('-n', '--num_epochs', default=10, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_name', default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100", "SVHN"],
                    help='set the dataset')
parser.add_argument('-l', '--learning_rate', default=1e-7, type=float, help='set the learning rate')
parser.add_argument('-N', '--batch_size', default=64, type=int, help='set the mini-batch size')
parser.add_argument('-w', '--weight_decay', default=5, type=float, help='set the parameter of the gaussian prior')
parser.add_argument('-p', '--precondition_decay_rate', default=.95, type=float,
                    help='set the decay rate of rescaling of the preconditioner')
parser.add_argument('-b', '--t_burn_in', default=0, type=int, help='set the burn in period')
parser.add_argument('-t', '--thinning', default=1, type=int, help='set the thinning')
parser.add_argument('--save_samples', default='True', help='if True we save the samples')
parser.add_argument('-g', '--ngpu', default=1, type=int, help='setl the number of gpus')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
args = parser.parse_args()

# Define the title to save the results
title = str()
for key, value in {'d_': args.dataset_name, 'l_': args.learning_rate, 'precondition_decay_rate_': args.precondition_decay_rate}.items():  # 'n_': args.num_epochs, 'b_': args.t_burn_in,
    title += '-' + key + str(value)


# Define the title to save the results
title = str()
for key, value in {'d_': args.dataset_name, 'm_': args.model, 'p_': args.proportion, 'c_': args.prob_update_param,
                   'l_': args.learning_rate}.items():
    title += '-' + key + str(value)

# Print the local torch version
print(f"\nTorch Version {torch.__version__}")

print("\nos.path.abspath(__file__) =\n\t%s\n" % os.path.abspath(__file__))
path = os.path.abspath('..')

# Save the path to store the data
path_workdir = '/gpfs/workdir/22-fed_avg-neurips/' + args.method + '_results/' if '/gpfs/usersv' in path else '/home/cloud/workdir/' + args.method + '_results/'
path_dataset = path_workdir + '../../dataset/'
# path_workdir = 'C:/Users/v84176953/Documents/Travaux_Actuels/' if os.path.exists(
#     'C:/Users/v84176953/Documents/Travaux_Actuels/') else './'  # TODO: comment here!
# path_dataset = path_workdir + '../dataset/'  # todo: enlever que pour l'ordinateur professionnel.
path_figures = path_workdir + 'figures/'
path_variables = path_workdir + 'variables/'
path_stats = path_variables + args.method + title
path_txt = path_variables + args.method + '_text' + title + '.txt'
path_save_samples = path_variables + 'samples-' + args.method + title

# Create the directory if it does not exist
save_samples = strtobool(args.save_samples)
os.makedirs(path_dataset, exist_ok=True)
os.makedirs(path_variables, exist_ok=True)
if save_samples:
    os.makedirs(path_save_samples, exist_ok=True)

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

# Load the saved networks
pretrained = False  # todo: change

# Define the network
if args.dataset_name == 'MNIST':
    model_dict = {'logistic': 'LogisticMnist', 'lenet5': 'LeNet5Mnist'}
    models = import_module('models.' + args.model)
    model_cfg = getattr(models, model_dict[args.model])
    net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
    transform = model_cfg.transform
elif args.dataset_name == 'CIFAR10':
    net = get_model(args.model, pretrained=pretrained)
    # Define the transformation
    normalize = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize)
    ])

# Number of worker to distribute the data
num_clients = 20  # todo: remettre 20.
max_data_size = np.inf  # todo: args.max_data_size if args.max_data_size == np.inf else len(trainset)

# Define the parameter of the dataset
params_train = {"root": path_dataset, "train": True, "transform": transform, "download": True}
params_test = {"root": path_dataset, "train": False, "transform": transform}

# Define the datasets
trainset = dataset(**params_train)
# We only consider a subset of the dataset if max_data_size < len(trainset)
if max_data_size < len(trainset):
    trainset = torch.utils.data.Subset(trainset, np.random.choice(len(trainset), max_data_size, replace=False))
split_size = len(trainset) // num_clients + (1 if len(trainset) % num_clients != 0 else 0)
trainloader_init = DataLoader(trainset, split_size, shuffle=False)
if strtobool(args.heterogeneous):
    print("\n--- Generate heterogeneous datasets ---\n")
    trainloader_init = heterogeneous_dataset(trainloader_init, num_clients, args.proportion)

# Modify the shape of the data to save time during the training stage
inputs = None
targets = None
length = 0
for data in trainloader_init:
    x = torch.unsqueeze(data[0], dim=1)
    y = torch.unsqueeze(data[1], dim=1)
    if inputs is None:
        length = len(y)
        inputs, targets = x, y
    else:
        length = min(length, len(y))
        inputs = torch.cat((inputs[:length], x[:length]), dim=1)
        targets = torch.cat((targets[:length], y[:length]), dim=1)

print('torch.unique(targets):', torch.unique(targets))  # TODO: suppr

# Define the loader
trainset = TensorDataset(inputs, targets)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True)

# Define the testloader
batch_size_test = 500
testset = dataset(**params_test)
testloader = DataLoader(testset, batch_size_test, shuffle=False)  # TODO: remettre testset

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# We start from the last sample
last_checkpoint = False  # todo: change here to load the last sample
epoch_init = -1
t_burn_in = args.t_burn_in
if last_checkpoint:
    fmax = -1
    for f in os.listdir(path_save_samples):
        if 'client' in f:
            continue
        if int(f) > fmax:
            fmax = int(f)
    if fmax > -1:
        epoch_init = fmax
        t_burn_in = args.thinning
        print(f'--- Load the last checkpoint: {fmax} ---\n ')
        net.load_state_dict(torch.load(os.path.join(path_save_samples, str(fmax))))

# Handle multi-gpu if desired
net.to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    net = nn.DataParallel(net, list(range(args.ngpu)))

# todo: delete
from utils.tools_dl import accuracy_model

print('Accuracy Beginning:', accuracy_model(net, testloader, device, verbose=False))  # todo: delete

# Define the parameters of the SGLD optimizer
speudo_batches = len(trainset)  # / args.batch_size  # we multiply the gradient by this quantity
num_burn_in_steps = args.t_burn_in * (speudo_batches // args.batch_size + (speudo_batches % args.batch_size > 0))
params_optimizer = {"lr": args.learning_rate, "num_pseudo_batches": speudo_batches,
                    "precondition_decay_rate": args.precondition_decay_rate, "num_burn_in_steps": num_burn_in_steps}

# Define the Sgld sampler
model = Sgld(net)

# Run the SGLD algorithm
model_state_dict, save_stats = model.run(trainloader, testloader, args.num_epochs, args.weight_decay / speudo_batches,
                                         params_optimizer, t_burn_in, args.thinning, epoch_init,
                                         strtobool(args.save_samples), path_save_samples)

# End the timer
executionTime = time.time() - startTime
print("Execution time =", executionTime)

# Save some RAM
del inputs, targets, trainset, testset, trainloader_init, trainloader, testloader  # , model

# Store the scores
save_dict = vars(args)
save_dict["execution time"] = executionTime
torch.save(save_stats, path_stats)

# Compute the scores
burn_in_preds = args.t_burn_in  # todo: change
run_scores(net, dataset, save_dict, title, path_figures, path_stats, path_txt, path_dataset, path_save_samples,
           device, transform, params_test, burn_in_preds, args)
