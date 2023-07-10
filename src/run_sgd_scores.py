#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
from importlib import import_module

import numpy as np
import proplot as pplt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import load_dataset
from utils.metrics import agreement, total_variation_distance
from utils.tools_dl import predictions
from utils.toy_tools_data import fusion, pprint
from utils.uncertainties_tools import accuracy_confidence, BS, calibration_curve, confidence, ECE, NLL

print("os.path.abspath(__file__) =\n\t", os.path.abspath(__file__))
path = os.path.abspath('..')

# Print the local torch version
print(f"Torch Version {torch.__version__}")

# Save the user choice of settings 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="resnet20_cifar10", help='set the model name')
parser.add_argument('-n', '--num_epochs', default=10, type=int, help='set the number of epochs')
parser.add_argument('-d', '--dataset_in', default="MNIST", choices=["MNIST", "CIFAR10", "CIFAR100"],
                    help='set the dataset')
parser.add_argument('--dataset_out', default="FashionMNIST")
parser.add_argument('-l', '--learning_rate', default=0.1, type=float, help='set the learning rate')
parser.add_argument('-N', '--batch_size', default=128, type=int, help='set the mini-batch size')
parser.add_argument('-t', '--tau', default=1., type=float,
                    help='percentage of data coming from the training distribution')
parser.add_argument('-g', '--ngpu', default=1, type=int, help='setl the number of gpus')
args = parser.parse_args()

title = str()
for key, value in {'m_': args.model, 'n_': args.num_epochs, 'l_': np.round(args.learning_rate, 3),
                   'N_': args.batch_size, 't_': args.tau}.items():
    title += '-' + key + str(value)

# Save the path to store the data
if os.path.isdir('/gpfs/workdir/'):
    path_workdir = f'/gpfs/workdir/22-fed_avg-neurips/{args.dataset_in}/'
elif os.path.isdir('/mnt/beegfs/'):
    path_workdir = f'/mnt/beegfs/workdir/avg-neurips/{args.dataset_in}/'
elif os.path.isdir('/home/cloud/home/'):
    path_workdir = f'/home/cloud/workdir/fedavg_aistats/{args.dataset_in}/'
else:
    path_workdir = './'
path_dataset = path_workdir + '../../dataset/'
path_figures = path_workdir + f'figures/'
path_variables = path_workdir + f'sgd{title.split("-t_")[0]}/'
path_samples = path_variables + f"samples/"
path_txt = path_variables + f"{args.dataset_out}.txt"
path_stats = path_variables + f'{args.dataset_out}{title.split("-t_")[1]}'

# Start the timer
startTime = time.time()

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
print(device)

# Define the network
model_dict = {'logistic': 'LogisticMnist', 'lenet5': 'LeNet5Mnist', 'resnet20': 'Resnet20Cifar10', 'mlp': 'MLPMnist'}
models = import_module('models.' + args.model)
model_cfg = getattr(models, model_dict[args.model])
if args.model == 'resnet20':
    model_cfg.kwargs = args.pretrained_model
transform = model_cfg.transform
net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
net.to(device)

# Define the transformation
pprint('--- Preparing data ---')

# Load the dataset function
if args.dataset_out == 'CIFAR10':  # TODO: there is a transform issue in this case !
    print('--- Load the dataset')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = getattr(load_dataset, "load_" + args.dataset_out)  # needed to compute agreement and total variation

    with open(path_dataset + '/cifar10_probs.csv', 'r') as fp:
        reference = np.loadtxt(fp)

else:
    dataset = getattr(torchvision.datasets, args.dataset_out)

# Define the parameter of the dataset
params_test = {"root": path_dataset, "transform": transform, "train": False, "download": True}

# Load the datasets
batch_size = 500
testset = dataset(**params_test)
testloader1 = DataLoader(testset, batch_size, shuffle=False)

# Load the In distribution dataset
dataset = getattr(torchvision.datasets, args.dataset_in)
testset = dataset(**params_test)
testloader0 = DataLoader(testset, batch_size, shuffle=False)

# Merge the datasets
testloader = fusion(testloader0, testloader1, args.tau)
ytest = testloader.dataset.tensors[1].numpy()

# # Load the targets  # TODO: not sure if it is relevant
# ytest = np.loadtxt(path_dataset + '/cifar10_test_y.csv').astype(
#     int) if args.dataset_out == 'CIFAR10' else testset.targets.numpy()
# testset.labels if dataset_in != "MNIST" else testset.targets  # TODO: vérifier

# For the statistics
save_dict = {}
tau_list = np.linspace(0, 1, num=100)

# Todo: maybe copy issue with dict.fromkeys(["final_acc", "agreement", ...], list)
save_dict["final_acc"] = []
save_dict["ece"] = []
save_dict["bs"] = []
save_dict["nll"] = []
if args.dataset_out == 'CIFAR10':
    save_dict["agreement"] = []
    save_dict["total_variation_distance"] = []

pprint("--- Compute the scores ---")

for it, f in enumerate(os.listdir(path_samples)):

    path = os.path.join(path_samples, f)
    print(f'Sample path: {f}')

    # Compute the predictions
    all_probs = predictions(testloader, net, path).cpu().numpy()
    preds = np.argmax(all_probs, axis=1)
    save_dict["final_acc"].append(100 * np.mean(preds == ytest))

    print('--- Final accuracy =', np.round(save_dict["final_acc"][-1], 1))

    # Load the HMC reference predictions
    if args.dataset_out == 'CIFAR10':
        # Now we can compute the metrics
        method_agreement = agreement(all_probs, reference)
        method_total_variation_distance = total_variation_distance(all_probs, reference)

        # Save the metrics
        save_dict["aggrement"].append(method_agreement)
        save_dict["total_variation_distance"].append(method_total_variation_distance)

        # Print the scores
        print("Agreement =", method_agreement, "Total variation =", method_total_variation_distance)

    # Compute the accuracy in function of p(y|x)>tau
    accuracies, misclassified = confidence(ytest, all_probs, tau_list)

    # Compute the Expected Calibration Error (ECE)
    save_dict["ece"].append(ECE(all_probs, ytest, num_bins=20))

    # Compute the Brier Score
    save_dict["bs"].append(BS(ytest, all_probs))

    # Compute the accuracy - confidence
    acc_conf = accuracy_confidence(all_probs, ytest, tau_list, num_bins=20)

    # Compute the calibration curve
    cal_curve = calibration_curve(all_probs, ytest, num_bins=20)

    # Compute the Negative Log Likelihood (NLL)
    save_dict["nll"].append(NLL(all_probs, ytest))

pprint('--- Save the results ---')

# Write the results
with open(path_txt, 'w') as f:
    f.write('\t---' + "SGD".upper() + '---\n\n')
    for key, value in save_dict.items():
        f.write('%s: %s / %s\n' % (key, np.mean(value), np.std(value)))

# Save the statistics
save_dict["ytest"] = ytest
save_dict["all_probs"] = all_probs
save_dict["tau_list"] = tau_list
save_dict["accuracies"] = accuracies
save_dict["calibration_curve"] = cal_curve
save_dict["accuracy_confidence"] = acc_conf

# Save the dictionary
if os.path.exists(path_stats):
    saved_dict = torch.load(path_stats)
    save_dict.update(saved_dict)
torch.save(save_dict, path_stats)

pprint('--- Display the results ---')

# Display the results
fig, axs = pplt.subplots(ncols=2, refwidth=2)
axs[0].plot(tau_list, acc_conf)
axs[0].format(title="Confidence in function of the accuracy")
axs[1].plot(cal_curve[1], cal_curve[0] - cal_curve[1])
axs[1].format(title="Accuracy - Confidence in function of the confidence")
axs.format(grid=True)
fig.legend(frameon=True)
fig.savefig(path_figures + f"sgd{title}-{args.dataset_out}.pdf", bbox_inches='tight')
