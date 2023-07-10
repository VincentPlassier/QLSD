#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset


def load_CIFAR10(root, train = False, transform = torch.Tensor):
    if train:
        path_x = os.path.join(root, "cifar10_train_x.csv")
        path_y = os.path.join(root, "cifar10_train_y.csv")
    else:
        path_x = os.path.join(root, "cifar10_test_x.csv")
        path_y = os.path.join(root, "cifar10_test_y.csv")

    x = np.loadtxt(path_x)
    x = x.reshape((len(x), 32, 32, 3))
    x_stack = []
    for xx in map(transform, x):
        x_stack.append(xx)
    x = torch.stack(x_stack).float()
    y = np.loadtxt(path_y).astype(int)
    return TensorDataset(x.reshape((len(x), 3, 32, 32)), torch.from_numpy(y))


# from jax import numpy as jnp


def load_CIFAR10_jax(root, train = False, transform = lambda x: x):
    # todo : modify this function
    if train:
        path_x = os.path.join(root, "cifar10_train_x.csv")
        path_y = os.path.join(root, "cifar10_train_y.csv")
    else:
        path_x = os.path.join(root, "cifar10_test_x.csv")
        path_y = os.path.join(root, "cifar10_test_y.csv")

    x = np.loadtxt(path_x)  # , delimiter=","
    x = x.reshape((len(x), 32, 32, 3))
    x_stack = []
    for xx in map(transform, x):
        x_stack.append(xx)
    x = torch.stack(x_stack).float()
    y = np.loadtxt(path_y).astype(int)  # , delimiter=","
    x = jnp.array(x.numpy(), dtype=jnp.float32)
    return x, y


from torch.utils.data import DataLoader


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        # todo: fix the issue - ValueError: only one element tensors can be converted to Python scalars
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):

    def __init__(self, dataset, batch_size = 1,
                 shuffle = False, sampler = None,
                 batch_sampler = None, num_workers = 0,
                 pin_memory = False, drop_last = False,
                 timeout = 0, worker_init_fn = None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))
