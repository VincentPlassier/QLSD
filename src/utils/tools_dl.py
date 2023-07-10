import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@torch.no_grad()
def accuracy_model(model, loader, device = "cuda:0", verbose = False):
    if verbose:
        loader = tqdm(loader)
    # To disable Dropout and BatchNormalization
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():  # useless normally
        for images, labels in loader:
            # to run on gpu
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


@torch.no_grad()
def predictions(loader, model, path, device = "cuda:0", verbose = False):
    model.to(device)
    # Load the parameters and network state
    model.load_state_dict(torch.load(path))
    if verbose:
        loader = tqdm(loader)
    all_probs = None
    for inputs, targets in loader:
        # get log probs
        log_probs = F.log_softmax(model(inputs.to(device)), dim=1)
        # get preds
        probs = torch.exp(log_probs)
        all_probs = probs if all_probs is None else torch.cat((all_probs, probs))
    return all_probs


def client_solver(data, model, num_epochs = 10, batch_size = 64, lr = .001, momentum_decay = 0.9, weight_decay = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainset = TensorDataset(data[0], data[1])
    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    # Optimizer parameters
    grad_clip = .1
    model.train()
    # Define the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum_decay, weight_decay=weight_decay)
    scheduler = OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(trainloader))
    #
    for epoch in range(num_epochs):
        loss_mean = 0.
        total, correct = 0, 0
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
            # update the average loss
            loss_mean = (i * loss_mean + loss.item()) / (i + 1)
            # update the accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # print statistics
        # if i % 165 == 164:  # print the statistics every 100 mini-batches
        print(f"Accuracy = {np.round(100 * correct / total, 1)}%,", 'Mean loss = %.3f' % loss_mean)
    return model.state_dict()
