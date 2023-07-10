#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.sgld_tools import SGLD, accuracy_model
from torch.optim.lr_scheduler import CyclicLR


class Sgld:

    def __init__(self, net):
        #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduction='mean')  # 'sum'
        # Store the statistics
        self.save_dict = {"count_grad": 0, "count_eval_grad": [0], "count_bits": 0, "store_bits": [0],
                          "losses_test": [], "accuracies_test": [], "mse_relative": []}

    def net_update(self, trainloader, epoch, weight_decay):
        # Optimizer parameters
        # grad_clip = .1
        self.net.train()
        loss_mean = 0.0
        total, correct = 0, 0
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            for param in self.net.parameters():
                loss += weight_decay * torch.norm(param) ** 2
            loss.backward()
            # nn.utils.clip_grad_value_(self.net.parameters(), grad_clip)
            self.optimizer.step()
            # update the learning rate
            self.scheduler.step()
            # update the average loss
            loss_mean = (i * loss_mean + loss.item()) / (i + 1)
            # update the accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print statistics
            if i % 165 == 164:  # print the statistics every 100 mini-batches
                print(f"Accuracy = {np.round(100 * correct / total, 1)}%,", 'Mean loss = %.3f' % loss_mean)
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                # add the number of communicated bits
                self.save_dict["count_bits"] += 32 * (i + 1) * len(torch.flatten(param))
        # save the number of evaluated gradients
        self.save_dict["count_eval_grad"].append(total)
        # save the bits communicated
        self.save_dict["store_bits"].append(self.save_dict["count_bits"])
        # print the statistics
        print('--- Train --- Epoch %d, Accuracy = %.2f%%, Loss = %.3f\n' % (
            epoch + 1, 100 * correct / total, (i + 1) * loss_mean))

    def save_results(self, testloader, epoch, t_burn_in, thinning, save_samples, path_save_samples):
        # calculate some statistics
        acc_test = accuracy_model(self.net, testloader, self.device)
        # add the new predictions with the previous ones
        if epoch >= t_burn_in and (
                epoch - t_burn_in) % thinning == 0 and save_samples and path_save_samples is not None:
            torch.save(self.net.state_dict(), path_save_samples + '/%s' % epoch)
        # save the accuracy
        self.save_dict["accuracies_test"].append(acc_test)
        # print the statistics
        print("--- Test --- Epoch: {}, Test accuracy: {}\n".format(epoch + 1, acc_test))

    def run(self, trainloader, testloader, num_iter, weight_decay = 5, params_optimizer = None, t_burn_in = 0,
            thinning = 1, epoch_init = -1, save_samples = False, path_save_samples = None):
        # define the optimizer
        self.optimizer = SGLD(self.net.parameters(), **params_optimizer)
        self.scheduler = CyclicLR(self.optimizer, params_optimizer["lr"], params_optimizer["lr"], step_size_up=2000,
                                  cycle_momentum=False)
        for epoch in range(epoch_init + 1, epoch_init + 1 + num_iter):
            self.net_update(trainloader, epoch, weight_decay)
            self.save_results(testloader, epoch, t_burn_in, thinning, save_samples, path_save_samples)
        return self.net.state_dict(), self.save_dict
