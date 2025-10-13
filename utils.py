import torch
import numpy as np

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(a, device) for a in x]
    return x.to(device)

def normalize(x, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    # If single-channel MNIST, mean/std will be scalars or length 1
    if x.size(1) == 1:
        mean = (0.1307,)
        std = (0.3081,)
    mean = torch.tensor(mean).view(1,x.size(1),1,1).to(x.device)
    std  = torch.tensor(std).view(1,x.size(1),1,1).to(x.device)
    return (x - mean) / std

def unnormalize(x, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    if x.size(1) == 1:
        mean = (0.1307,)
        std = (0.3081,)
    mean = torch.tensor(mean).view(1,x.size(1),1,1).to(x.device)
    std  = torch.tensor(std).view(1,x.size(1),1,1).to(x.device)
    return x * std + mean

def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()
