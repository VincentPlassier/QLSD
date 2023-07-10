import torch
import torch.nn as nn
from torchvision import transforms


def MLP(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 100),
        nn.ReLU(),
        # nn.Dropout(p=p),
        nn.Linear(100, output_dim),
    )


class MLPMnist:
    base = MLP
    args = list()
    kwargs = {"input_dim": 784, "output_dim": 10}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
