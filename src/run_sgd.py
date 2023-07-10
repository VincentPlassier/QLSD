#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
from importlib import import_module

import numpy as np
import proplot as pplt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from torch._C import default_generator
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.tools_dl import accuracy_model
from utils.toy_tools_data import pprint

print("os.path.abspath(__file__) =\n\t", os.path.abspath(__file__))
path = os.path.abspath('..')

# Print the local torch version
print(f"Torch Version {torch.__version__}")

# Save the user choice of settings 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="resnet20_cifar10", help='set the model name')
parser.add_argument('-n', '--num_epochs', default=10, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_name', default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100", "SVHN"],
                    help='set the dataset')
parser.add_argument('-l', '--learning_rate', default=0.1, type=float, help='set the learning rate')
parser.add_argument('--scheduler_method', default='OneCycleLR', help='set the scheduler method')
parser.add_argument('-N', '--batch_size', default=128, type=int, help='set the mini-batch size')
parser.add_argument('-g', '--ngpu', default=1, type=int, help='setl the number of gpus')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
parser.add_argument('-r', '--resume', default=-1, type=int, help='resume from checkpoint')
args = parser.parse_args()

title = str()
for key, value in {'m_': args.model, 'n_': args.num_epochs, 'l_': np.round(args.learning_rate, 3),
                   'N_': args.batch_size, 's_': args.seed}.items():
    title += '-' + key + str(value)

# Save the path to store the data
if os.path.isdir('/gpfs/workdir/'):
    path_workdir = f'/gpfs/workdir/22-fed_avg-neurips/{args.dataset_name}/'
elif os.path.isdir('/mnt/beegfs/'):
    path_workdir = f'/mnt/beegfs/workdir/22-fed_avg-neurips/{args.dataset_name}/'
elif os.path.isdir('/home/cloud/home/'):
    path_workdir = f'/home/cloud/workdir/fedavg_aistats/{args.dataset_name}/'
else:
    path_workdir = './'
path_dataset = path_workdir + '../../dataset/'
path_figures = path_workdir + f'figures/'
path_variables = path_workdir + f'sgd{title.split("-s_")[0]}/'
path_samples = path_variables + f"samples/"

# Create the directory if it does not exist
os.makedirs(path_dataset, exist_ok=True)
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_samples, exist_ok=True)

# If true, we save the samples
save_weights = True

# If true, we save the .npy
save_npy = True

# Set random seed for reproducibility
seed_np = args.seed if args.seed != -1 else None
seed_torch = args.seed if args.seed != -1 else default_generator.seed()
np.random.seed(seed_np)
torch.manual_seed(seed_torch)
torch.cuda.manual_seed(seed_torch)

# Start the timer
startTime = time.time()

# Define the network
model_dict = {'logistic': 'LogisticMnist', 'lenet5': 'LeNet5Mnist', 'resnet20': 'Resnet20Cifar10', 'mlp': 'MLPMnist'}
models = import_module('models.' + args.model)
model_cfg = getattr(models, model_dict[args.model])
if args.model == 'resnet20':
    model_cfg.kwargs = args.pretrained_model
net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
transform = model_cfg.transform

# Define the transformation
pprint('--- Preparing data ---')

# Load the function associated with the chosen dataset
dataset = getattr(torchvision.datasets, args.dataset_name)

if args.dataset_name == 'CIFAR10':  # TODO: MAYBE IT BIASED THE RESULTS
    # Image augmentation is used to train the model
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Only the data is normalized we do not need to augment the test data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train, transform_test = transform, transform

# Define the parameter of the dataset
params_train = {"root": path_dataset, "transform": transform_train, "train": True, "download": True}
params_test = {"root": path_dataset, "transform": transform_test, "train": False}

# Load the datasets
batch_size_test = 500
trainset = dataset(**params_train)
testset = dataset(**params_test)
trainloader = DataLoader(trainset, args.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size_test, shuffle=False)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# Handle multi-gpu if desired
net.to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    net = nn.DataParallel(net, list(range(args.ngpu)))


# Initialized the weights
def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.weight.data.fill_(0.01)
        # m.bias.data.fill_(0.01)


net.apply(init_weights)

# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)


# init_params(net)

if args.resume > -1:
    print('Resume from a previous checkpoint')
    print(path_samples + str(args.resume))
    net.load_state_dict(torch.load(path_samples + str(args.resume)))
    accuracy = accuracy_model(net, testloader, device, verbose=True)
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)


# To count the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


print('Number of parameters for one neural network is: %s.' % count_parameters(net))


def sgd_solver(trainloader, testloader, model, num_epochs = 10, momentum = 0.9, lr = .001,
               scheduler_method = 'OneCycleLR'):
    # Save the log
    log = {'loss_train': [], 'accuracy_train': [], 'accuracy_test': []}
    # Optimizer parameters
    grad_clip = .1
    model.train()
    # Define the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler_method == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(trainloader))
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    # Start the optimization procedure
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            # update the learning rate
            scheduler.step()
            # compute the accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            # update the running loss
            running_loss += loss.item()
            # print statistics
            if i % 150 == 149:  # print the statistics every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        log['loss_train'].append(running_loss)
        log['accuracy_train'].append(100 * correct / total)
        log['accuracy_test'].append(accuracy_model(model, testloader, device))
    print('\n--- Finished Training ---\n')
    acc = accuracy_model(model, testloader, device, verbose=True)
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)
    return model.state_dict(), acc, log


pprint("--- Run SGD ---")

# Run the SGD algorithm
model_state_dict, accuracy_test, log = sgd_solver(trainloader, testloader, net, args.num_epochs, lr=args.learning_rate,
                                                  scheduler_method=args.scheduler_method)

# End the timer
executionTime = time.time() - startTime

# Set the palette
sns.set_palette("colorblind")

# Save the results
count = 0
while os.path.isfile(path_variables + f"{count}.txt"):
    count += 1
save_dict = vars(args)
save_dict["execution time"] = executionTime
save_dict["accuracy_test"] = accuracy_test
save_dict["log"] = log
with open(path_variables + f"{count}.txt", 'w') as f:
    f.write('\t--- SGD ---\n\n')
    for key, value in save_dict.items():
        f.write('%s: %s\n' % (key, value))
torch.save(model_state_dict, path_samples + str(count))

pprint('--- Display the results ---')

# Set the color
pplt.rc.cycle = '538'

# Plot the results
fig, ax = pplt.subplots()
for key, item in log.items():
    ax.plot(item, marker='o', ms=1, markevery=10, markeredgecolor='k', markeredgewidth=1, label=key)

ax.set_yscale('log')
ax.format(grid=True, xlabel=r"Number of epoch", ylabel=r"Accuracy/Loss", fontsize=12)
fig.legend(loc='b',
           fontsize=10,
           prop={'size': 10},
           order='C',
           title_fontsize=12,
           frameon=True,
           shadow=True,
           facecolor='white',
           edgecolor='k',
           labelspacing=1,
           handlelength=2)

fig.savefig(os.path.join(path_figures, f'sgd_stats{title}.pdf'), bbox_inches='tight')

pprint('--- Save the results ---')
