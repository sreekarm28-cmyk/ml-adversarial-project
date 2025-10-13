import torch
import torch.nn.functional as F

from utils import normalize, unnormalize

def clamp(X, lower_limit, upper_limit):
    return torch.clamp(X, min=lower_limit, max=upper_limit)

def fgsm(model, x, y, eps=8/255, device='cpu'):
    # x is unnormalized [0,1] or normalized depending on caller; here we assume x passed is model-ready
    model.eval()
    x_orig = x.detach()
    x_adv = x_orig.clone().detach().requires_grad_(True)

    out = model(x_adv)
    loss = F.cross_entropy(out, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + eps * grad.sign()
    return x_adv.detach()

def pgd(model, x, y, eps=8/255, alpha=2/255, iters=10, device='cpu', clip_min=0.0, clip_max=1.0):
    """PGD operating in pixel space (x in [0,1] raw pixel). Model expects normalized input inside."""
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = clamp(x_adv, clip_min, clip_max)

    for i in range(iters):
        x_adv.requires_grad_(True)
        x_adv_norm = normalize(x_adv)
        logits = model(x_adv_norm)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = clamp(x_adv, x_orig - eps, x_orig + eps)
        x_adv = clamp(x_adv, clip_min, clip_max)
    return x_adv.detach()

def pgd_pipeline(pipeline_fn, x, y, eps=8/255, alpha=2/255, iters=10, clip_min=0.0, clip_max=1.0):
    """PGD attack where pipeline_fn accepts unnormalized [0,1] x and returns logits (handles normalization)."""
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = clamp(x_adv, clip_min, clip_max)
    for i in range(iters):
        x_adv.requires_grad_(True)
        out = pipeline_fn(x_adv)
        loss = F.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = clamp(x_adv, x_orig - eps, x_orig + eps)
        x_adv = clamp(x_adv, clip_min, clip_max)
    return x_adv.detach()

# Optional: foolbox wrappers for CW & DeepFool if available
try:
    import foolbox as fb
    FOOLBOX_AVAILABLE = True
except Exception:
    FOOLBOX_AVAILABLE = False

def cw_l2(model, x, y, device='cpu'):
    if not FOOLBOX_AVAILABLE:
        raise RuntimeError('Foolbox is not installed; CW attack unavailable.')
    fmodel = fb.PyTorchModel(model, bounds=(0,1))
    attack = fb.attacks.L2CarliniWagnerAttack()
    imgs = x.cpu().numpy()
    labels = y.cpu().numpy()
    advs = attack(fmodel, imgs, labels, epsilons=[0.3])
    return torch.tensor(advs).to(device)

def deepfool(model, x, y, device='cpu'):
    if not FOOLBOX_AVAILABLE:
        raise RuntimeError('Foolbox is not installed; DeepFool attack unavailable.')
    fmodel = fb.PyTorchModel(model, bounds=(0,1))
    attack = fb.attacks.DeepFoolAttack()
    imgs = x.cpu().numpy()
    labels = y.cpu().numpy()
    advs = attack(fmodel, imgs, labels)
    return torch.tensor(advs).to(device)
