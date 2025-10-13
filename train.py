import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_loaders
from models import get_resnet18, get_mnist_cnn
from utils import normalize, unnormalize, to_device, accuracy
from attacks import pgd

def train_baseline(args):
    device = args.device
    train_loader, test_loader = get_loaders(batch_size=args.batch_size, dataset=args.dataset, num_workers=args.num_workers)
    if args.dataset == 'mnist':
        model = get_mnist_cnn(num_classes=args.num_classes).to(device)
    else:
        model = get_resnet18(num_classes=args.num_classes, in_channels=(1 if args.dataset=='mnist' else 3)).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            # if dataset is MNIST and x has 1 channel, model handles it
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        # eval
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred==y).sum().item(); total += x.size(0)
        acc = correct/total
        print(f"Epoch {epoch}: Test Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'baseline.pth')
    print(f"Best Acc: {best_acc}")

def train_adv(args):
    device = args.device
    train_loader, test_loader = get_loaders(batch_size=args.batch_size, dataset=args.dataset, num_workers=args.num_workers)
    if args.dataset == 'mnist':
        model = get_mnist_cnn(num_classes=args.num_classes).to(device)
    else:
        model = get_resnet18(num_classes=args.num_classes, in_channels=(1 if args.dataset=='mnist' else 3)).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"AdvTrain Epoch {epoch}")
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            # x is normalized in dataloader; convert to pixel space [0,1]
            x_unnorm = unnormalize(x)
            x_adv = pgd(model, x_unnorm, y, eps=args.eps, alpha=args.alpha, iters=args.pgd_steps, device=device)
            # normalize adv back
            x_adv_norm = normalize(x_adv)
            logits = model(x_adv_norm)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        # eval on clean
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred==y).sum().item(); total += x.size(0)
        acc = correct/total
        print(f"Epoch {epoch}: Test Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'adv_trained.pth')
    print(f"Best Acc: {best_acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline','adv'], default='baseline')
    parser.add_argument('--dataset', choices=['cifar','mnist'], default='cifar')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--pgd_steps', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()

    if args.mode == 'baseline':
        train_baseline(args)
    else:
        train_adv(args)
