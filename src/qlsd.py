#!/usr/bin/env python
# coding: utf-8

import copy
import torch
import numpy as np
import torch.nn as nn
from utils.sgld_tools import accuracy_model
from utils.compression_algo import StochasticQuantization


class Qlsd:

    def __init__(self, net, num_workers, svrg_epoch, compression_parameter, use_memory_term = True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # Define the compression operator
        self.quantization_fun = StochasticQuantization(compression_parameter)
        #
        self.net = net
        net_state_dict = copy.deepcopy(net.state_dict())
        # Store the statistics
        self.save_dict = {"count_grad": 0, "count_eval_grad": [0], "count_bits": 0, "store_bits": [0],
                          "accuracies_test": []}
        # Define net_svrg, grad_svrg required for the variance reduce scheme
        self.svrg_epoch = svrg_epoch
        if self.svrg_epoch != np.infty:
            # create the network net_svrg
            self.net_svrg = copy.deepcopy(net)
            self.net_svrg.to(self.device)
            # initialize the variance reduction parameter
            self.grad_svrg = dict()
            for i in range(num_workers):
                self.grad_svrg[i] = {}
                for name, param in net_state_dict.items():
                    self.grad_svrg[i][name] = torch.zeros_like(param)
        # Define the memory term
        self.memory_term = dict()
        # initialize the memory term
        for i in range(num_workers):
            self.memory_term[i] = {}
            for name, param in net_state_dict.items():
                self.memory_term[i][name] = param if use_memory_term else torch.zeros_like(param)
        # Define the memory rates : if memory_rate = 0, there is no memory term
        self.memory_rate = dict()
        for name, param in net_state_dict.items():
            if use_memory_term and compression_parameter > 0:
                alpha = np.sqrt(len(torch.flatten(param))) / compression_parameter
                self.memory_rate[name] = 1 / (min(alpha, alpha ** 2) + 1)
            else:
                self.memory_rate[name] = 0

    def net_update(self, trainloader, epoch, weight_decay, lr, mini_batch_sizes, num_participants):
        net_state_dict = copy.deepcopy(self.net.state_dict())
        self.net.train()
        # initialize the sum of the compressed gradients
        grad_sum = dict()
        for name, param in net_state_dict.items():
            grad_sum[name] = torch.zeros_like(param)
        # set the clients which communicate
        participating_clients = np.random.choice(len(mini_batch_sizes), num_participants, replace=False)
        # each participating worker compute its gradient in parallel
        for i, data_i in filter(lambda x: True if x[0] in participating_clients else False, enumerate(trainloader)):
            pseudo_batches = len(data_i[1])
            # sample the mini-batch
            if epoch % self.svrg_epoch == 0 and self.svrg_epoch != np.infty:
                batch = data_i
            else:
                # the i-th worker samples a mini-batch of size N_i
                ind_batch = np.random.choice(len(data_i[1]), size=mini_batch_sizes[i], replace=False)
                batch = data_i[0][ind_batch], data_i[1][ind_batch]
            # zero the parameter gradients
            self.net.zero_grad()
            if epoch % self.svrg_epoch != 0 and self.svrg_epoch != np.infty:
                self.net_svrg.zero_grad()
            # update the number of calculated gradients
            self.save_dict["count_grad"] += len(batch[1])
            # compute the loss
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            for param in self.net.parameters():
                loss += weight_decay / pseudo_batches * torch.norm(param) ** 2
            # compute the gradient of the loss with respect to all Tensors with requires_grad=True
            loss.backward()
            # compute the loss associated with the net_svrg
            if epoch % self.svrg_epoch != 0 and self.svrg_epoch != np.infty:
                loss_svrg = self.criterion(self.net_svrg(inputs), labels)
                for param in self.net_svrg.parameters():
                    loss_svrg += weight_decay / (2 * pseudo_batches) * torch.norm(param) ** 2
                loss_svrg.backward()
            # disable the gradient calculation
            with torch.no_grad():
                for k, (name, param) in enumerate(self.net.named_parameters()):
                    if epoch % self.svrg_epoch == 0 and self.svrg_epoch != np.infty:
                        # save the i-th worker gradient
                        self.grad_svrg[i][name] = pseudo_batches * param.grad.data
                        # compute the gradient transmitted by the i-th worker
                        transmited_grad = self.grad_svrg[i][name] - self.memory_term[i][name]
                    elif self.svrg_epoch != np.infty:
                        g_svrg = list(self.net_svrg.parameters())[k].grad.data
                        # compute the gradient transmitted by the i-th worker
                        transmited_grad = pseudo_batches * (param.grad.data - g_svrg) + \
                                          self.grad_svrg[i][name] - self.memory_term[i][name]
                    else:
                        # the worker computes a gradient estimate and subtracts its memory term
                        transmited_grad = pseudo_batches * param.grad.data - self.memory_term[i][name]
                    # add the number of communicated bits
                    self.save_dict["count_bits"] += self.quantization_fun.communication_eval(transmited_grad)
                    # quantized the difference between the grad and the memory term
                    delta_grad = self.quantization_fun.quantize(transmited_grad)
                    # add the i-th worker gradient
                    grad_sum[name] += self.quantization_fun.quantize(delta_grad) + self.memory_term[i][name]
                    # update the memory term
                    self.memory_term[i][name] += self.memory_rate[name] * delta_grad
        # Parameter updates - perform the QLSD step
        for (name, param) in self.net.named_parameters():
            if param.grad is None:
                continue
            noise = torch.sqrt(2 / lr) * torch.normal(mean=torch.zeros_like(param), std=torch.ones_like(param))
            scaled_grad = (len(mini_batch_sizes) / num_participants) * grad_sum[name] + noise
            param.data.add_(-lr * scaled_grad)
        # Update the svrg network
        if epoch % self.svrg_epoch == 0 and self.svrg_epoch != np.infty:
            self.net_svrg = copy.deepcopy(self.net)
        # save the number of evaluated gradients
        self.save_dict["count_eval_grad"].append(self.save_dict["count_grad"])
        # save the bits communicated
        self.save_dict["store_bits"].append(self.save_dict["count_bits"])

    def save_results(self, testloader, epoch, t_burn_in, thinning, save_samples, path_save_samples):
        # save the sampled weights
        if epoch >= t_burn_in and (
                epoch - t_burn_in) % thinning == 0 and save_samples and path_save_samples is not None:
            torch.save(self.net.state_dict(), path_save_samples + '/%s' % epoch)
            # calculate some statistics
            acc_test = accuracy_model(self.net, testloader, self.device)
            # save the accuracy
            self.save_dict["accuracies_test"].append(acc_test)
            # print the statistics
            print("--- Test --- Epoch: {}, Test accuracy: {}\n".format(epoch + 1, acc_test))

    def run(self, trainloader, testloader, num_iter, weight_decay = 5, lr = 1e-5, mini_batch_sizes = None,
            num_participants = None, t_burn_in = 0, thinning = 1, epoch_init = -1, save_samples = False,
            path_save_samples = None):
        num_participants = min(num_participants, len(mini_batch_sizes))
        for epoch in range(1 + epoch_init, epoch_init + 1 + num_iter):
            self.net_update(trainloader, epoch, weight_decay, torch.Tensor([lr]).to(self.device), mini_batch_sizes,
                            num_participants)
            self.save_results(testloader, epoch, t_burn_in, thinning, save_samples, path_save_samples)
        return self.net.state_dict(), self.save_dict
