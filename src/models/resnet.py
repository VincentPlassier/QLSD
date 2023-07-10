import torch
from pytorchcv.model_provider import get_model
from torchvision import transforms


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Resnet20Cifar10:
    def base(name, pretrained = False): return get_model(name, pretrained=pretrained).apply(weight_init)

    args = ['resnet20_cifar10']
    kwargs = {'pretrained': False}

    normalize = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4, fill=128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*normalize)
    ])


class Resnet20Cifar100:
    def base(name, pretrained = False): return get_model(name, pretrained=pretrained).apply(weight_init)

    args = ['resnet20_cifar100']
    kwargs = {'pretrained': False}

    normalize = [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)]
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4, fill=128),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(*normalize)
    ])
