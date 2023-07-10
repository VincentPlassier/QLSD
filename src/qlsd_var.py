#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import itertools
import time
import os
import argparse
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch._C import default_generator

# Print the local torch version
print(f"Torch Version {torch.__version__}")

# Save the path to store the data
path_workdir = '/workdir/21-federated_results'
path = path_workdir + '/fashion_mnist_experiment/qlsd_var_results'
path_theta = path + '/theta_samples'

# Create the directory if it does not exist
if not os.path.exists(path_workdir + '/dataset'):
    os.makedirs(path_workdir + '/dataset')
if not os.path.exists(path_theta):
    os.makedirs(path_theta)

# Save the user choice of settings 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='BNN', choices=["BNN", "CNN", "Net"], help='set the model name')
parser.add_argument('-n', '--num_iter', default=100, type=int, help='set the number of epochs')
parser.add_argument('--ngpu', default=1, type=int, help='set the number of gpus')
parser.add_argument('-g', '--gamma', default=1e-05, type=float, help='set the learning rate')
parser.add_argument('-l', '--l2_reg', default=10, type=float, help='set the parameter of the gaussian prior')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set the seed')
args = parser.parse_args()

# If true, we save the samples
save_weights = False

# If true, we save the .npy
save_npy = True

# SVRG parameter: if epoch_svrg = np.infty, there is no variance reduction scheme
epoch_svrg = 5

# Define the compression parameter
compression_parameter = 2 ** 8

# Number of update for theta
num_iter = args.num_iter

# Number of worker to distribute the data
num_workers = 100

# Mini-batch size of each worker
N = 120 * np.ones(num_workers, dtype=int)

# Set random seed for reproducibility
seed_np = args.seed if args.seed != -1 else None
seed_torch = args.seed if args.seed != -1 else default_generator.seed()
np.random.seed(seed_np)
torch.manual_seed(seed_torch)
torch.cuda.manual_seed(seed_torch)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = args.ngpu

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# Define the step size
print(args.gamma)
gamma = torch.tensor([args.gamma]).to(device)

# Start the timer
startTime = time.time()


## Load the dataset


"""Dataset"""


# define the classes
classes = list(range(0, 10))  # [c0, c1]

# Load the datasets
trainset = torchvision.datasets.MNIST(root=path_workdir + '/dataset', train=True,
                                      download=True)
testset = torchvision.datasets.MNIST(root=path_workdir + '/dataset', train=False,
                                     download=True)  # train = False

# Normalize the dataset
X_train, Y_train = trainset.data, trainset.targets
X_test, Y_test = testset.data, testset.targets
X_train, X_test = X_train.float() / 255, X_test.float() / 255
x_mean, x_std = X_train.mean(), X_train.std()
X_train, X_test = (X_train - x_mean) / x_std, (X_test - x_mean) / x_std
X_train, X_test = X_train.unsqueeze(dim=1), X_test.unsqueeze(dim=1)


class DataLoader:

    def __init__(self, my_iter, data_size):
        self.it = 0
        self.my_iter = itertools.cycle(my_iter)
        self.data_size = data_size

    def __iter__(self):
        self.it = 0
        return self

    def __next__(self):
        if self.it < self.data_size:
            self.it += 1
            return next(self.my_iter)
        else:
            self.it = 0
            raise StopIteration

    next = __next__

    def __len__(self):
        len_ = 0
        for j in range(self.data_size):
            len_ += len(next(self.my_iter)[1])
        return len_


def torch_split(tensor, num_sections=1):
    remainder = len(tensor) % num_sections
    div_points = len(tensor) // num_sections * np.arange(num_sections + 1)
    if remainder != 0:
        div_points[- remainder:] += np.arange(1, remainder + 1, dtype=np.int16)
    sub_tensor = []
    for i in range(num_sections):
        start = div_points[i]
        end = div_points[i + 1]
        sub_tensor.append(tensor[start: end])
    return sub_tensor


# Define an iterator based on the split dataset
iter_train = zip(torch_split(X_train, num_workers), torch_split(Y_train, num_workers))
num_test_split = 5
iter_test = zip(torch_split(X_test, num_test_split), torch_split(Y_test, num_test_split))

# Define the train and test loader
trainloader = DataLoader(iter_train, num_workers)
testloader = DataLoader(iter_test, num_test_split)

# Shapes of the train datasets
print('Shapes of the training datasets:')
for i, z in enumerate(trainloader):
    print('\t', z[0].shape, z[1].shape)

# Shape of the test dataset
print('\nShape of the testing dataset:')
for i, z in enumerate(testloader):
    print('\t', z[0].shape, z[1].shape)


class LoadBatch():
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset[1])
    
    def mini_batch(self, batch_size):
        ind_batch = np.random.choice(self.data_size, size=batch_size, replace=False)
        x_batch, y_batch = self.dataset[0][ind_batch], self.dataset[1][ind_batch]
        return x_batch, y_batch
    
    def sample_batch(self, num_batchs=1, batch_size=1):
        X_batch, Y_batch = [], []
        for i in range(num_batchs):
            x, y = self.mini_batch(batch_size)
            X_batch.append(x); Y_batch.append(y)
        return zip(X_batch, Y_batch)
    
    def sample_all(self, num_batchs=1):
        return zip(torch_split(self.dataset[0], num_batchs), torch_split(self.dataset[1], num_batchs))


# Define trainloader_split to easily sample mini-batches
trainloader_split = []
for (i, dataset_i) in enumerate(trainloader):
    trainloader_split.append(LoadBatch(dataset_i))

# One example:
loader = trainloader_split[0]
dataiter = loader.sample_batch(num_batchs=2, batch_size=4)
for x, y in dataiter:
    print(y)


# function to show an image
def imshow(img):
    npimg = img.numpy()
    npimg = (npimg - np.min(npimg)) / (np.max(npimg) - np.min(npimg))
    plt.imshow(npimg.transpose(1, 2, 0))
    plt.axis(False)

# get some random training images
dataiter = loader.sample_batch(num_batchs=1, batch_size=4)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images[:4]))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


## Algorithme de compression


class StochasticQuantization:

    def __init__(self, s=1):
        if s == 0:
            print("There will be no compression")
            # raise ValueError(("There will be no compression")
        self.s = s

    def Qs(self, v):
        if self.s == 0:
            return v.to(device)
        v_norm = torch.norm(v, p=2)
        if v_norm == 0:
            return v.to(device)
        r = self.s * torch.abs(v) / v_norm
        l = torch.floor(r)
        l += torch.ceil(r - l) - torch.ones_like(l)
        b = torch.bernoulli(r - l)
        xi = (l + b) / self.s
        return (v_norm * torch.sign(v) * xi).to(device)
    
    def communication_eval(self, v):
        s = self.s
        dim_v = len(torch.flatten(v))
        if s == 0:            
            return 32 * dim_v  # there is no quantization
        elif s < np.sqrt(dim_v / 2 - np.sqrt(dim_v)):
            t = s * (s + np.sqrt(dim_v))
            return 32 + 3 * t * (1 +  np.log(2 * (s ** 2 + dim_v) / t) / 2)
        else:
            t = s ** 2 + min(dim_v, s * np.sqrt(dim_v))
            return 32 + dim_v * (2 + (1 + np.log(1 + t / dim_v)) / 2)


# Test the unbiasness of the compressor operator
compression_operator = StochasticQuantization(compression_parameter)
nb_mcmc = 1000
v = torch.tensor([[1.7, 0.5], [3.4, 4.2]])
torch_mean = torch.zeros_like(v).to(device)
for i in range(nb_mcmc):
    torch_mean += compression_operator.Qs(v)
print(torch.norm(v.to(device) - torch_mean / nb_mcmc))


## Neural Network


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return torch.softmax(self.fc2(x), dim=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)  # F.log_softmax(self.fc3(x), dim=1)


""" Neural Network"""


class BNN(nn.Module):

    def __init__(self):
        super(BNN, self).__init__()
        # fully connected layer
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1) # F.log_softmax(self.fc2(x), dim=1) --> output: log(p(y=i|x))


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        # torch.nn.init.xavier_uniform_(m.weight)
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)


# Define the network
model_dict = {"BNN": BNN(), "CNN": CNN(), "Net": Net()}
net = model_dict[args.model]
net.to(device)

# Initialized the weights
net.apply(init_weights)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    net = nn.DataParallel(net, list(range(ngpu)))

# To count the total number of parameters
def count_parameters(net):
    return sum(p.numel() for p in net.parameters())

print('Number of parameters for one neural network is: %s.' % count_parameters(net))


## Training Stage


"""Initialization of the training stage"""


# Define the loss function
criterion = nn.CrossEntropyLoss(reduction='sum')

# L2 penalization corresponding to a gaussian prior
l2_reg = args.l2_reg

# Define the compression operator
quantization_fun = StochasticQuantization(compression_parameter)

# Count the number of gradient evaluations
count_grad = 0

# Store the total number of gradient evaluations
count_eval_grad = [0]

# Count the number of gradient evaluations
count_bits = 0

# Store the total number of communicated bits
store_bits = [0]

# Will contain the losses
losses_test = []

# Will contain the accuracies
accuracies_test = []

# Will contain the sampled parameters
theta_net = []

# Will contains the predicted probabilities for each sample
prob_all = []

# Define the file title
title = '/seed={0}-worker={1}-num_iter={2}-N={3}-gamma={4:.1E}'.format(seed_torch, num_workers, num_iter, N[0], gamma[0])

# Define iter_save which contains the epochs for which we want to save the parameters
iter_save = [0, 1]
tau = .515
while iter_save[-1] < num_iter:
    new_el = tau * np.sum(iter_save[-2:])
    iter_save.append(new_el)
iter_save = np.unique(np.round(iter_save[:-1] + [num_iter]))

# Initialize the sum of the compressed gradients
grad_sum = dict()

# Define net_svrg, grad_svrg required for the variance reduce scheme
if epoch_svrg != np.infty:
    # create the network net_svrg
    net_svrg = copy.deepcopy(net)
    # initialize the variance reduction parameter
    grad_svrg = dict()
    for i in range(num_workers):
        grad_svrg[i] = {}
        for name, param in net.named_parameters():
            grad_svrg[i][name] = torch.zeros_like(param)

# Define the memory term
memory_term = dict()
# initialize the memory term
for i in range(num_workers):
    memory_term[i] = {}
    for name, param in net.named_parameters():
        memory_term[i][name] = torch.zeros_like(param)

# Define the memory rates : if memory_rate = 0, there is no memory term
memory_rate = dict()
for name, param in net.named_parameters():
    if compression_parameter > 0:
        alpha = np.sqrt(len(torch.flatten(param))) / compression_parameter
        memory_rate[name] = 1 / (min(alpha, alpha ** 2) + 1)
    else:
        memory_rate[name] = 0

# Save statistics before the training stage
correct = 0
total = 0
loss_test = 0

with torch.no_grad():
    # compute the prior
    # for param in net.parameters():
    #     loss_test += l2_reg * torch.norm(param) ** 2
    for images, labels in testloader:
        # to run on gpu
        images, labels = images.to(device), labels.to(device)
        # compute the predictions
        outputs = net(images)
        # predict the most likely label
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss_test += criterion(outputs, labels).item()
# save the loss
losses_test.append(loss_test)  # .cpu().numpy())
# Save the accuracy
accuracies_test.append(100 * correct / total)
# Print the loss and accuracy
print(losses_test[-1], accuracies_test[-1])

# for epoch in range(num_iter):
for epoch in range(num_iter):
    
    # initialize the sum of the compressed gradients
    for name, param in net.named_parameters():
        grad_sum[name] = torch.zeros_like(param)
    
    # each worker compute its gradient in parallel
    for i, trainloader_i in enumerate(trainloader_split):
        
        # sample the mini-batch
        if epoch % epoch_svrg == 0 and epoch_svrg != np.infty:
            mini_batchs = trainloader_i.sample_all(num_batchs=1)
        else:
            # the i-th worker samples a mini-batch of size N_i
            mini_batchs = trainloader_i.sample_batch(num_batchs=1, batch_size=N[i])
        
        # zero the parameter gradients
        net.zero_grad()
        if epoch % epoch_svrg != 0 and epoch_svrg != np.infty:
            net_svrg.zero_grad()
        
        # initialized the losses
        loss, loss_svrg = 0, 0
        
        # compute the loss over the mini-batch
        for images, labels in mini_batchs:
                    
            # update the number of calculated gradients
            count_grad += len(labels)
        
            # to run on gpu
            images, labels = images.to(device), labels.to(device)
            
            # compute the predictions()
            outputs = net(images)
            
            # update the loss
            loss += criterion(outputs, labels)
            
            # compute the loss associated with the net_svrg
            if epoch % epoch_svrg != 0 and epoch_svrg != np.infty:
                outputs_svrg = net_svrg(images)
                loss_svrg += criterion(outputs_svrg, labels)
        
        # add the gaussian prior
        for name, param in net.named_parameters():
            loss += l2_reg / num_workers * torch.norm(param) ** 2
            if epoch % epoch_svrg != 0 and epoch_svrg != np.infty:
                loss_svrg += l2_reg / num_workers * torch.norm(net_svrg.state_dict()[name]) ** 2
        
        # compute the gradient of loss with respect to all Tensors with requires_grad=True
        loss.backward()
        if epoch % epoch_svrg != 0 and epoch_svrg != np.infty:
            loss_svrg.backward()
        
        # disable the gradient calculation
        with torch.no_grad():
            for k, (name, param) in enumerate(net.named_parameters()):
                if epoch % epoch_svrg == 0 and epoch_svrg != np.infty:
                    # save the i-th worker gradient
                    grad_svrg[i][name] = param.grad.data
                    # compute the gradient transmitted by the i-th worker
                    transmited_grad = param.grad.data - memory_term[i][name]
                elif epoch_svrg != np.infty:
                    g_svrg = list(net_svrg.parameters())[k].grad.data
                    # compute the gradient transmitted by the i-th worker
                    transmited_grad = trainloader_i.data_size / N[i] * (param.grad.data - g_svrg) + grad_svrg[i][name] - memory_term[i][name]
                else:
                    # the worker computes a gradient estimate and substracts its memory term
                    transmited_grad = trainloader_i.data_size / N[i] * param.grad.data - memory_term[i][name]            
                # add the number of communicated bits
                count_bits += quantization_fun.communication_eval(transmited_grad)
                # quantized the difference between the grad and the memory term
                delta_grad = quantization_fun.Qs(transmited_grad)
                # add the i-th worker gradient
                grad_sum[name] += quantization_fun.Qs(delta_grad) + memory_term[i][name]
                # update the memory term
                memory_term[i][name] += memory_rate[name] * delta_grad
    
    # Parameter updates
    with torch.no_grad():
        for name, param in net.named_parameters():
            # perform the LQSD step
            param.copy_(param - gamma * grad_sum[name] + torch.sqrt(2 * gamma) * torch.randn(param.shape).to(device))
    
    # Update the svrg network
    if epoch % epoch_svrg == 0 and epoch_svrg != np.infty:
        net_svrg = copy.deepcopy(net)
    
    # condition to save the predictions
    save_pred = (epoch > .1 * num_iter)
    
    # print the statistics and save the parameter theta
    if epoch + 1 in iter_save or save_pred:
        # add the iteration number in iter_save
        iter_save = np.hstack((iter_save, epoch + 1))
        # save the parameter theta
        if save_pred:
            # save the sampled parameter
            theta_net.append(copy.deepcopy(net))
        # define variables to compute the loss and the accuracy
        correct = 0
        total = 0
        loss_test = 0
        # to save the predictions of the test set
        prob_pred = []
        with torch.no_grad():
            # compute the prior
            # for param in net.parameters():
            #     loss_test += l2_reg * torch.norm(param) ** 2
            for images, labels in testloader:
                # to run on gpu
                images, labels = images.to(device), labels.to(device)
                # compute the predictions
                outputs = net(images)
                # predict the most likely label
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_test += criterion(outputs, labels).item()
                # add the new predictions with the previous ones
                if len(prob_pred) == 0:
                    prob_pred = outputs.detach().cpu().numpy()
                else:
                    prob_pred  = np.vstack((prob_pred, outputs.detach().cpu().numpy()))
        prob_all.append(prob_pred)
        # save the number of evaluated gradients
        count_eval_grad.append(count_grad)
        # save the bits communicated
        store_bits.append(count_bits)
        # save the loss
        losses_test.append(loss_test)  # .cpu().numpy())
        # save the accuracy
        accuracies_test.append(100 * correct / total)
        print('Epoch number: %s; loss = %s; accuracy %d %%.' % (epoch + 1, loss_test, 100 * correct / total))

# Remove iterations already saved
iter_save = np.unique(iter_save)


## Save the results


if save_weights:
    for (i, net) in enumerate(theta_net):
        torch.save(net, path_theta + title + '_' + str(i + 1))
        # Another possibility to save the parameters
        # torch.save(net.module.state_dict(), path_theta + title + '_' + str(i + 1))        

# Display the loss in function of the number of gradient evaluations
fig, ax = plt.subplots()
ax.plot(count_eval_grad, losses_test)
ax.set(xlabel='Number of gradient evaluations', ylabel='Loss')
ax.set_yscale('log')
ax.grid()
# fig.savefig(path + '/losses_grad_eval-' + title[1:] + '.pdf', bbox_inches='tight')

# Display the loss in function of the number of communicated_bits
fig, ax = plt.subplots()
ax.plot(store_bits, losses_test)
ax.set(xlabel='Number of communicated bits', ylabel='Loss')
ax.set_yscale('log')
ax.grid()
# fig.savefig(path + '/losses_bits-' + title[1:] + '.pdf', bbox_inches='tight')

# Display the loss in function of the epoch
fig, ax = plt.subplots()
ax.plot(iter_save, losses_test)
ax.set(xlabel='Iteration', ylabel='Loss')
ax.set_yscale('log')
ax.grid()
fig.savefig(path + '/losses_test-' + title[1:] + '.pdf', bbox_inches='tight')

# Display the accuracy in function of the number of gradient evaluations
fig, ax = plt.subplots()
ax.plot(count_eval_grad, accuracies_test)
ax.set(xlabel='Number of gradient evaluations', ylabel='Accuracy')
ax.set_yscale('log')
ax.grid()
# fig.savefig(path + '/accuracies_grad_eval-' + title[1:] + '.pdf', bbox_inches='tight')

# Display the accuracy in function of the number of communicated bits
fig, ax = plt.subplots()
ax.plot(store_bits, accuracies_test)
ax.set(xlabel='Number of communicated bits', ylabel='Accuracy')
ax.set_yscale('log')
ax.grid()
# fig.savefig(path + '/accuracies_bits-' + title[1:] + '.pdf', bbox_inches='tight')

# Display the accuracy in function of the epoch
fig, ax = plt.subplots()
ax.plot(iter_save, accuracies_test)
ax.set(xlabel='Iteration', ylabel='Accuracy')
ax.set_yscale('log')
ax.grid()
fig.savefig(path + '/accuracies_test-' + title[1:] + '.pdf', bbox_inches='tight')

# Save the results
file = open(path + title + '.txt', 'w')
file.write(title[1:])
file.write('\nnum_iter = %s, \nnum_workers = %s, \nN = %s, \ngamma = %s, \nepoch_svrg = %s, \ncompression_parameter = %s, \nl2_reg = %s, ' % (num_iter, num_workers, N[0], gamma[0], epoch_svrg, compression_parameter, l2_reg))
file.write('\n\nNetwork: %s' %net)
file.write('\n\n\nQLSD performances: \nloss = %s, \naccuracy = %s, \ncount_bits = %s, \ncount_grad = %s' % (losses_test[-1], accuracies_test[-1], count_bits, count_grad))
file.close()  # to change the file access mode


## Uncertainty quantifications


class PostNet:

    def __init__(self, theta_net):
        if len(theta_net) == 0:
            raise ValueError("theta_net must be non-empty.") 
        self.theta_net = theta_net

    def __call__(self, x):
        with torch.no_grad():
            m = 0
            for net in self.theta_net:
                m += net(x).detach()
            m /= len(self.theta_net)
        return m

    def predict(self, x, k=1):
        return torch.argsort(self(x), dim=-1, descending=True)[:, :k]


# Load the predicted probabilities
# prob_all = np.load(path + '/prob_all-' + title[1:] + '.npy')

# # Create a list to contain the saved networks
# theta_net = []

# # Load the saved networks
# for name in os.listdir(path_theta):
#     if title[1:] not in name:
#         continue
#     net = torch.load(path_theta + '/' + name)
#     net.to(device)
#     theta_net.append(net)

# Define the network obtained by sampling according to the posterior
post_net = PostNet(theta_net)

# Select some test images
images_set = testloader.next()[0][:16]

# Displays the selected images
fig, ax = plt.subplots()
imshow(torchvision.utils.make_grid(images_set))
# fig.savefig(path + '/test_images.pdf', bbox_inches='tight')


def pred_uncertainties(x, net_list, alpha=.05):
    y_pred = np.zeros((len(net_list), len(classes)))
    for i, net in enumerate(net_list):
        y_pred[i] = net(x).detach().data.numpy()    
    predictions = np.mean(y_pred, axis=0)  # or np.median
    label_pred = np.argmax(predictions)
    prob_pred = predictions[label_pred]    
    return classes[label_pred], y_pred[:, label_pred]


# Compute the accuracy in function of p(y|x)>tau
tau_list = np.linspace(0, 1, num=100)
accuracies = np.zeros_like(tau_list) + 1
predictions = np.mean(prob_all, axis=0)  # or np.median
ind_pred = np.argmax(predictions, axis=1)
prob_pred = np.max(predictions, axis=1)
label_pred = np.take_along_axis(np.array(classes), ind_pred, axis=0)
misclassified = np.where(label_pred != Y_test.numpy())[0]
for (i, tau) in enumerate(tau_list):
    ind_tau = np.where(prob_pred > tau)[0]
    if len(ind_tau) == 0:
        tau_list = tau_list[:i]
        accuracies = accuracies[:i]
    else:
        ind_inter = np.intersect1d(ind_tau, misclassified)
        accuracies[i] = 1 - len(ind_inter) / len(ind_tau)


# Display the accuracy in function of the confidence parameter: tau
fig, ax = plt.subplots()
ax.plot(tau_list, accuracies)
ax.set(xlabel='tau', ylabel='Accuracy')
ax.grid()
fig.savefig(path + '/confidence-' + title[1:] + '.pdf', bbox_inches='tight')
plt.clf()

# Compute the Expected Calibration Eror (ECE)
acc = np.ones_like(prob_pred)
acc[misclassified] = 0
ECE = np.mean(np.abs(acc - prob_pred))
print("ECE =", ECE)

# Perform a one-hot encoding
labels_true = np.eye(len(classes))[Y_test]
# Compute the Brier Score (BS)
BS = np.mean((predictions - labels_true) ** 2)
print("BS =", BS)

# Compute the Negative Log Likelihood (NLL)
entropy_mnist = - np.log(np.take_along_axis(predictions, np.expand_dims(Y_test.numpy(), axis=1), axis=1)).squeeze()
NLL = entropy_mnist.mean()
print(f"NLL = {NLL}")

# Load the FashionMNIST dataset
fmnist_testset = torchvision.datasets.FashionMNIST(root=path_workdir + '/dataset', train=True, download=True)  # train = False
Xfmnist_test, Yfmnist_test = fmnist_testset.data, fmnist_testset.targets

# Normalize the MNIST dataset
Xfmnist_test = Xfmnist_test.float() / 255
Xfmnist_test = (Xfmnist_test - x_mean) / x_std
Xfmnist_test = Xfmnist_test.unsqueeze(dim=1)

# Define the FashionMNIST test loader
iter_fmnist_test = zip(torch_split(Xfmnist_test, 5), torch_split(Yfmnist_test, 5))
loader_fmnist_test = DataLoader(iter_fmnist_test, 5)

# Compute the predicted probabilities
predicted_fmnist = list()
with torch.no_grad():
    for images, labels in loader_fmnist_test:
        images, labels = images.to(device), labels.to(device)
        outputs = post_net(images).cpu().numpy()
        if len(predicted_fmnist) == 0:
            predicted_fmnist = outputs
        else:
            predicted_fmnist  = np.vstack((predicted_fmnist, outputs))

# Compute the Negative Log Likelihood (NLL)
entropy_mnist = - np.log(np.take_along_axis(predictions, np.expand_dims(Y_test.numpy(), axis=1), axis=1)).squeeze()
NLL = entropy_mnist.mean()
print(f"NLL = {NLL}")

# Load the FashionMNIST dataset
fmnist_testset = torchvision.datasets.FashionMNIST(root=path_workdir + '/dataset', train=True, download=True)  # train = False
Xfmnist_test, Yfmnist_test = fmnist_testset.data, fmnist_testset.targets

# Normalize the MNIST dataset
Xfmnist_test = Xfmnist_test.float() / 255
Xfmnist_test = (Xfmnist_test - x_mean) / x_std
Xfmnist_test = Xfmnist_test.unsqueeze(dim=1)

# Define the FashionMNIST test loader
iter_fmnist_test = zip(torch_split(Xfmnist_test, 5), torch_split(Yfmnist_test, 5))
loader_fmnist_test = DataLoader(iter_fmnist_test, 5)

# Compute the predicted probabilities
predicted_fmnist = list()
with torch.no_grad():
    for images, labels in loader_fmnist_test:
        images, labels = images.to(device), labels.to(device)
        outputs = post_net(images).cpu().numpy()
        if len(predicted_fmnist) == 0:
            predicted_fmnist = outputs
        else:
            predicted_fmnist  = np.vstack((predicted_fmnist, outputs))

# Compute the Negative Log Likelihood (NLL) on MNIST
entropy_fmnist = - np.log(np.take_along_axis(predicted_fmnist, np.expand_dims(Yfmnist_test.numpy(), axis=1), axis=1)).squeeze()

# Display the predicted entropies
entropy_dict = {"Mnist": entropy_mnist, "Fashion Mnist": entropy_fmnist}
ax = sns.kdeplot(data=entropy_dict, fill=True, cut=0, common_norm=False)
ax.set(xlabel='pred. entropy',
       ylabel='Density')  # title='')
plt.savefig(path + '/Predicted entropy-' + title[1:] + '.pdf', bbox_inches='tight')

# Compute the Area Under the Curve (AUC)
AUC = 0
for y in entropy_mnist:
    AUC += np.sum(y <= entropy_fmnist)
AUC /= len(entropy_mnist) * len(entropy_fmnist)
print(f"AUC = {AUC}")

# Compute the execution time
executionTime = time.time() - startTime

# Getting the RAM memory
total_memory, used_memory, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
# Calculate the used RAM percentage
ram_percentage = np.round(100 * used_memory / total_memory, 2)
print("RAM memory {0:.0%} used.".format(used_memory / total_memory))

# Store the ECE, BS, NNL, AUC, the execution time
file = open(path + title + '.txt', 'a')
file.write(f"\nECE = {ECE}, \nBS = {BS}, \nNLL = {NLL}, \nAUC = {AUC}, \nTime = {executionTime}, \nRAM = {ram_percentage}")
file.close()  # to change the file access mode


# Save the results
if save_npy:
    # save the accuracies and the losses
    np.save(path + '/prob_all-' + title[1:], np.squeeze(prob_all))
    np.save(path + '/iter_save-' + title[1:], iter_save)
    np.save(path + '/count_eval_grad-' + title[1:], count_eval_grad)
    np.save(path + '/store_bits-' + title[1:], store_bits)
    np.save(path + '/losses_test-' + title[1:], losses_test)
    np.save(path + '/accuracies_test-' + title[1:], accuracies_test)
    # save the accuracy in function of p(y|x)>tau
    np.save(path + '/confidence-' + title[1:], accuracies)
    # save the entropies
    np.save(path + '/entropy_mnist-' + title[1:], entropy_mnist)
    np.save(path + '/entropy_fmnist-' + title[1:], entropy_fmnist)
