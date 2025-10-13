# Implements the black-box defenses: JPEG, Gaussian, KMeans, TVM (ROF), and VQ-VAE wrapper
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.cluster import KMeans

def jpeg_compress_batch(x, quality=90):
    # x: tensor batch in [0,1], shape (B,C,H,W)
    B = x.shape[0]
    out = []
    for i in range(B):
        img = TF.to_pil_image(x[i].cpu())
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=int(quality))
        buf.seek(0)
        img2 = Image.open(buf).convert('RGB' if x.shape[1]==3 else 'L')
        t = TF.to_tensor(img2)
        out.append(t)
    return torch.stack(out, dim=0).to(x.device)

def gaussian_smooth_batch(x, kernel_size=3, sigma=0.3):
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        out[i] = TF.gaussian_blur(x[i], kernel_size=kernel_size, sigma=sigma)
    return out

def kmeans_compress_batch(x, n_colors=16):
    # operates per-image
    B,C,H,W = x.shape
    out = torch.zeros_like(x)
    for i in range(B):
        img = (x[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8).reshape(-1, C)
        # sklearn KMeans expects 2D array
        kmeans = KMeans(n_clusters=n_colors, n_init=1, random_state=0).fit(img)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        labels = kmeans.labels_
        recon = centers[labels].reshape(H,W,C)
        recon = recon.astype(np.float32)/255.0
        out[i] = torch.from_numpy(recon).permute(2,0,1)
    return out.to(x.device)

def tv_denoise(img, weight=0.1, n_iter=50):
    # Rudin-Osher-Fatemi (ROF) denoising; img HxWxC numpy float
    u = img.copy()
    px = np.zeros_like(img)
    py = np.zeros_like(img)
    nm = 1.0 / 8.0
    for _ in range(n_iter):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        px_new = px + nm * ux
        py_new = py + nm * uy
        norm = np.maximum(1.0, np.sqrt(px_new**2 + py_new**2))
        px = px_new / norm
        py = py_new / norm
        div_p = (np.roll(px, 1, axis=1) - px) + (np.roll(py, 1, axis=0) - py)
        u = img + weight * div_p
    return u

def tvm_batch(x, weight=0.03, n_iter=60):
    # x: tensor batch in [0,1]
    out = torch.zeros_like(x)
    B = x.shape[0]
    for i in range(B):
        img = x[i].permute(1,2,0).cpu().numpy()
        den = tv_denoise(img, weight=weight, n_iter=n_iter)
        out[i] = torch.from_numpy(den).permute(2,0,1)
    return out.to(x.device)

def apply_vqvae_batch(vqvae, x, device='cpu'):
    # vqvae: instance of VQVAE that expects input in [0,1] and returns reconstruction
    vqvae.eval()
    with torch.no_grad():
        rec, qloss = vqvae(x.to(device))
    return rec
