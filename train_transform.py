import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_loaders
from models import get_resnet18, SimpleTransform, get_mnist_cnn
from utils import normalize, unnormalize
from attacks import pgd_pipeline

class PipelineWrapper(nn.Module):
    """Wrapper pipeline: takes unnormalized [0,1] input, applies transform (pixel space),
       then normalizes and forwards to classifier."""
    def __init__(self, transform_net, classifier):
        super().__init__()
        self.T = transform_net
        self.C = classifier
    def forward(self, x):
        x_t = self.T(x)
        x_t_norm = normalize(x_t)
        return self.C(x_t_norm)

def train_joint(args):
    device = args.device
    train_loader, test_loader = get_loaders(batch_size=args.batch_size, dataset=args.dataset, num_workers=args.num_workers)
    if args.dataset == 'mnist':
        classifier = get_mnist_cnn(num_classes=args.num_classes).to(device)
    else:
        classifier = get_resnet18(num_classes=args.num_classes, in_channels=(1 if args.dataset=='mnist' else 3)).to(device)
    transform_net = SimpleTransform().to(device)

    optimizer = optim.SGD(list(classifier.parameters()) + list(transform_net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    pipeline = PipelineWrapper(transform_net, classifier)
    best_acc = 0.0

    for epoch in range(args.epochs):
        pipeline.train(); transform_net.train(); classifier.train()
        pbar = tqdm(train_loader, desc=f"JointTrain Epoch {epoch}")
        for x_norm,y in pbar:
            x_norm, y = x_norm.to(device), y.to(device)
            # x_norm is normalized; get pixel [0,1]
            x_raw = unnormalize(x_norm)

            # craft x_adv in pixel space w.r.t pipeline
            def pipeline_fn(inp):
                return pipeline(inp)
            x_adv = pgd_pipeline(pipeline_fn, x_raw, y, eps=args.eps, alpha=args.alpha, iters=args.pgd_steps)

            # forward
            x_t = transform_net(x_raw)
            out_clean = classifier(normalize(x_t))
            x_t_adv = transform_net(x_adv)
            out_adv = classifier(normalize(x_t_adv))

            loss_cls = F.cross_entropy(out_clean, y)
            loss_adv = F.cross_entropy(out_adv, y)
            loss_recon = F.mse_loss(x_t, x_raw)
            loss = args.alpha_w * loss_cls + args.beta_w * loss_adv + args.gamma_w * loss_recon

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        # eval on clean
        classifier.eval(); transform_net.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x_norm,y in test_loader:
                x_norm,y = x_norm.to(device), y.to(device)
                x_raw = unnormalize(x_norm)
                x_t = transform_net(x_raw)
                out = classifier(normalize(x_t))
                pred = out.argmax(dim=1)
                correct += (pred==y).sum().item(); total += x_norm.size(0)
        acc = correct/total
        print(f"Epoch {epoch}: Test Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'transform': transform_net.state_dict(), 'classifier': classifier.state_dict()}, 'transform_joint.pth')
    print(f"Best Acc: {best_acc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar', choices=['cifar','mnist'])
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--pgd_steps', type=int, default=5)
    parser.add_argument('--alpha_w', type=float, default=1.0)
    parser.add_argument('--beta_w', type=float, default=1.0)
    parser.add_argument('--gamma_w', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    train_joint(args)
