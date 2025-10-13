import torchvision.transforms as T
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader

from utils import CIFAR10_MEAN, CIFAR10_STD

def get_loaders(batch_size=128, data_dir='./data', num_workers=0, dataset='cifar'):
    dataset = dataset.lower()
    if dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
        train_transform = T.Compose([
            T.Resize(32),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        test_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        train = MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test = MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    else:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        train = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test  = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
