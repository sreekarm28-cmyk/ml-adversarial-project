import torch
import torch.nn as nn
import torchvision.models as models

# -----------------------
# Vector Quantizer (VQ)
# -----------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs: (B, C, H, W) with C == embedding_dim expected
        input_shape = inputs.shape
        flat_input = inputs.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)

        # Compute distances
        # ||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2
        distances = (
            (flat_input**2).sum(dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
            + (self.embedding.weight.t()**2).sum(dim=0, keepdim=True)
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(
            input_shape[0], input_shape[2], input_shape[3], self.embedding_dim
        )
        quantized = quantized.permute(0,3,1,2).contiguous()

        # Losses
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss

# -----------------------
# VQ-VAE
# -----------------------
class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden=128, num_embeddings=512, embedding_dim=128):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, embedding_dim, 1, 1, 0)
        )
        self.quant = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, in_channels, 1, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        z_e = self.enc(x)
        z_q, qloss = self.quant(z_e)
        x_rec = self.dec(z_q)
        return x_rec, qloss

# -----------------------
# SimpleTransform (autoencoder-like)
# -----------------------
class SimpleTransform(nn.Module):
    """Lightweight encoder-decoder transform net. Input/Output in [0,1] space expected."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(inplace=True)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(32,3,3,1,1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out

# -----------------------
# ResNet18 for CIFAR
# -----------------------
def get_resnet18(num_classes=10, pretrained=False, in_channels=3):
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -----------------------
# MNIST small CNN (2 Conv + 2 FC) - matches report
# -----------------------
class MNISTCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_mnist_cnn(num_classes=10):
    return MNISTCNN(num_classes=num_classes)
