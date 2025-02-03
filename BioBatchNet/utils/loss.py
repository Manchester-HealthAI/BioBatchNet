import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # Moved bandwidth_multipliers to be a buffer (on the same device as the model)
        self.register_buffer("bandwidth_multipliers", mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        # Ensure L2_distances and bandwidth_multipliers are on the same device
        L2_distances = torch.cdist(X, X) ** 2
        bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)  # Ensure the correct device
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

def orthogonal_loss(z1, z2):
    z1_centered = z1 - z1.mean(dim=0, keepdim=True)
    z2_centered = z2 - z2.mean(dim=0, keepdim=True)
    cov = torch.matmul(z1_centered.t(), z2_centered) / (z1.size(0) - 1)
    ortho_loss = torch.norm(cov, p='fro') ** 2
    return ortho_loss

def l1_loss(model):
    l1 = 0.0
    for param in model.encoder2.parameters():
        l1 += torch.sum(torch.abs(param))  
    return l1

def kl_divergence(mu, logvar):
    kl = -0.5 * (torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return kl

def pairwise_loss(p1, p2, cons_type):
    if cons_type == "ML":
        ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
        return ml_loss
    else:
        cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
        return cl_loss
    
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-4
        if isinstance(scale_factor, float): 
            scale_factor = torch.tensor(scale_factor, dtype=torch.float32, device=mean.device).unsqueeze(0)
        mean = mean * scale_factor       
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result
