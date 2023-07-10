import torch
import torch.nn as nn
from torchvision import transforms


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
        # self.soft = nn.Softmax(dim=0)

    def forward(self, x):
        # return self.soft(self.lin(x))  # We do this step later
        return self.lin(x)


class LogisticMnist:
    base = LogisticRegression
    args = list()
    kwargs = {"input_dim": 784, "output_dim": 10}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
