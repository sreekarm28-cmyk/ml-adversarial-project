import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np

from datasets import get_loaders
from models import get_resnet18, VQVAE, SimpleTransform, get_mnist_cnn
from utils import normalize, unnormalize
from attacks import pgd, fgsm, pgd_pipeline
from defenses import jpeg_compress_batch, gaussian_smooth_batch, kmeans_compress_batch, tvm_batch, apply_vqvae_batch

try:
    import foolbox as fb
    FOOLBOX_AVAILABLE = True
except Exception:
    FOOLBOX_AVAILABLE = False

def eval_model(model, test_loader, device='cpu'):
    correct = 0; total = 0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred==y).sum().item(); total += x.size(0)
    return correct/total

def eval_under_pgd(model, test_loader, device='cpu', eps=8/255, alpha=2/255, steps=10):
    # model expects normalized inputs
    correct = 0; total = 0
    for x_norm,y in tqdm(test_loader, desc=f'PGD eval eps={eps}'):
        x_norm,y = x_norm.to(device), y.to(device)
        x_raw = unnormalize(x_norm)
        x_adv = pgd(model, x_raw, y, eps=eps, alpha=alpha, iters=steps, device=device)
        x_adv_norm = normalize(x_adv)
        out = model(x_adv_norm)
        pred = out.argmax(dim=1)
        correct += (pred==y).sum().item(); total += x_norm.size(0)
    return correct/total

def eval_defenses(transform, classifier, test_loader, device='cpu'):
    results = {}
    # clean accuracy
    clean_acc = 0; total = 0
    with torch.no_grad():
        for x_norm,y in test_loader:
            x_raw,y = unnormalize(x_norm).to(device), y.to(device)
            x_t = transform(x_raw)
            out = classifier(normalize(x_t))
            pred = out.argmax(dim=1)
            clean_acc += (pred==y).sum().item(); total += x_raw.size(0)
    results['clean'] = clean_acc/total

    keys = ['pgd_undef', 'pgd_jpeg', 'pgd_gauss', 'pgd_kmeans', 'pgd_tvm', 'pgd_vqvae']
    counters = {k:0 for k in keys}
    totals = {k:0 for k in keys}

    for x_norm,y in tqdm(test_loader, desc='Defenses PGD eval'):
        x_raw,y = unnormalize(x_norm).to(device), y.to(device)
        # craft adversarial w.r.t pipeline (transform+clf)
        def pipeline_fn(inp):
            x_t = transform(inp)
            return classifier(normalize(x_t))
        x_adv = pgd_pipeline(pipeline_fn, x_raw, y, eps=8/255, alpha=2/255, iters=5)

        # undefended classifier on adversarial
        out = classifier(normalize(x_adv))
        pred = out.argmax(dim=1)
        counters['pgd_undef'] += (pred==y).sum().item(); totals['pgd_undef'] += x_raw.size(0)

        # JPEG
        x_jpeg = jpeg_compress_batch(x_adv, quality=90)
        out = classifier(normalize(x_jpeg))
        counters['pgd_jpeg'] += (out.argmax(dim=1)==y).sum().item(); totals['pgd_jpeg'] += x_raw.size(0)

        # Gaussian
        x_gauss = gaussian_smooth_batch(x_adv, kernel_size=3, sigma=0.3)
        out = classifier(normalize(x_gauss))
        counters['pgd_gauss'] += (out.argmax(dim=1)==y).sum().item(); totals['pgd_gauss'] += x_raw.size(0)

        # KMeans
        x_kmean = kmeans_compress_batch(x_adv, n_colors=50 if x_raw.shape[1]==3 else 16)
        out = classifier(normalize(x_kmean))
        counters['pgd_kmeans'] += (out.argmax(dim=1)==y).sum().item(); totals['pgd_kmeans'] += x_raw.size(0)

        # TVM
        x_tvm = tvm_batch(x_adv, weight=0.03, n_iter=60)
        out = classifier(normalize(x_tvm))
        counters['pgd_tvm'] += (out.argmax(dim=1)==y).sum().item(); totals['pgd_tvm'] += x_raw.size(0)

        # VQ-VAE (if using VQVAE as transform)
        try:
            # If transform *is* a VQVAE instance, apply as defense on x_adv
            if hasattr(transform, 'enc') and isinstance(transform, VQVAE):
                x_vq = apply_vqvae_batch(transform, x_adv, device=device)
            else:
                # if transform is a SimpleTransform, we can also try to pass x_adv through transform again
                x_vq = transform(x_adv)
            out = classifier(normalize(x_vq))
            counters['pgd_vqvae'] += (out.argmax(dim=1)==y).sum().item(); totals['pgd_vqvae'] += x_raw.size(0)
        except Exception:
            pass

    for k in keys:
        results[k] = counters[k] / totals[k] if totals[k] > 0 else None
    return results

def eval_cw_deepfool(pipeline_fn, test_loader, device='cpu'):
    if not FOOLBOX_AVAILABLE:
        print("Foolbox not available — skipping CW & DeepFool")
        return None

    fmodel = fb.PyTorchModel(pipeline_fn, bounds=(0,1))
    attack_cw = fb.attacks.L2CarliniWagnerAttack()
    attack_df = fb.attacks.DeepFoolAttack()

    results = {"cw_acc":0.0, "df_acc":0.0, "total":0}
    for x_norm,y in tqdm(test_loader, desc='CW/DeepFool Eval'):
        x_raw, y = unnormalize(x_norm).to(device), y.to(device)
        imgs = x_raw.cpu().numpy()
        labels = y.cpu().numpy()
        adv_cw = attack_cw(fmodel, imgs, labels, epsilons=[0.3])[0]
        adv_df = attack_df(fmodel, imgs, labels)[0]
        x_cw = torch.tensor(adv_cw).to(device)
        x_df = torch.tensor(adv_df).to(device)
        out_cw = pipeline_fn(x_cw)
        out_df = pipeline_fn(x_df)
        pred_cw = out_cw.argmax(dim=1)
        pred_df = out_df.argmax(dim=1)
        results["cw_acc"] += (pred_cw==y).sum().item()
        results["df_acc"] += (pred_df==y).sum().item()
        results["total"] += x_raw.size(0)

    results["cw_acc"] /= results["total"]
    results["df_acc"] /= results["total"]
    return results

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # default dataset: cifar ; you may change to 'mnist'
    train_loader, test_loader = get_loaders(batch_size=128, dataset='cifar', num_workers=0)

    # load transform+classifier (attempt)
    transform = SimpleTransform().to(device)
    classifier = get_resnet18().to(device)
    try:
        ckpt = torch.load('transform_joint.pth', map_location=device)
        transform.load_state_dict(ckpt['transform'])
        classifier.load_state_dict(ckpt['classifier'])
    except Exception:
        print("Warning: could not load transform_joint.pth; using randomly initialized weights")

    transform.eval(); classifier.eval()

    print("Running PGD-based defense evaluation...")
    results = eval_defenses(transform, classifier, test_loader, device=device)
    print(pd.DataFrame([results]).T)

    # pipeline for foolbox: expects a function returning logits for normalized pipeline
    def pipeline_fn(inp):
        # inp is unnormalized [0,1]
        x_t = transform(inp)
        return classifier(normalize(x_t))

    if FOOLBOX_AVAILABLE:
        print("Running CW & DeepFool (via Foolbox). This is slow on CPU...")
        extra = eval_cw_deepfool(pipeline_fn, test_loader, device=device)
        print(extra)
    else:
        print("Foolbox not available; CW & DeepFool skipped.")
