import torch
import torch.nn as nn

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = -ctx.alpha * grad_outputs
        return grad_outputs, None
    
class GRL(nn.Module):
    def __init__(self, alpha) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x) -> torch.tensor:
        return GradientReversalLayer.apply(x, self.alpha)

#=======================================================================================================================#
class ResidualBlock(nn.Module):
    """
    Args:
    - int_sz : input size of firsr layer
    - out_sz_list: hidden layers and output layers
    - use_residual: whether use residual link
    - use_drop: Boolean to indicate if dropout should be used.
    - dropout: Dropout rate, defaults to 0.3.
    - use_bn: Boolean to indicate if batch normalization should be used.

    """
    def __init__(self, in_sz: int, out_sz: int, use_residual: bool=True, use_drop: bool=True, dropout: float=0.3, use_bn: bool=True) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.layer = nn.Sequential(nn.Linear(in_sz, out_sz),
                                   nn.BatchNorm1d(out_sz) if use_bn else nn.Identity(),
                                   nn.ReLU(),
                                   nn.Dropout(dropout) if use_drop else nn.Identity())
        
        self.residual = nn.Linear(in_sz, out_sz) if in_sz != out_sz else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return self.layer(x) +  self.residual(x)
        else:
            return self.layer(x)


class BaseStack(nn.Module):
    def __init__(self, input_sz:int, layer_sizes:list, use_residual:bool=True, use_drop:bool=True, dropout:float=0.3, use_bn:bool=True) -> None:
        super().__init__()

        # Initialize residual blocks directly with parameters
        self.layers = nn.ModuleList()
        sizes = [input_sz] + layer_sizes

        for i in range(1, len(sizes)):
            self.layers.append(ResidualBlock(sizes[i - 1], sizes[i], use_residual=use_residual, use_drop=use_drop, dropout=dropout, use_bn=use_bn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BaseEncoder(BaseStack):
    """
    Args:
    - in_sz: Input size.
    - hidden_layers: List of hidden layer sizes.
    - latent_sz: Size of the latent vector.
    - use_vae: Boolean indicating if VAE-style reparameterization is used.
    - use_drop: Boolean indicating if dropout should be used.
    - dropout: Dropout rate.
    - use_bn: Boolean indicating if batch normalization should be used.
    """
    
    def __init__(self, 
                 in_sz:int, 
                 hidden_layers:list, 
                 latent_sz:int, 
                 use_vae:bool=True, 
                 use_residual:bool=True, 
                 use_drop:bool=True, 
                 dropout:float=0.3,
                 use_bn:bool=True) -> None:
        """
        choose whether need residual and other functions
        """
        super().__init__(in_sz, hidden_layers + [latent_sz], use_residual=use_residual, use_drop=use_drop, dropout=dropout, use_bn=use_bn)
        
        self.use_vae = use_vae

        if use_vae: 
            self.mu_encoder = nn.Linear(latent_sz, latent_sz)
            self.logvar_encoder = nn.Linear(latent_sz, latent_sz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
            
        if self.use_vae:
            mu = self.mu_encoder(x)
            logvar = self.logvar_encoder(x)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        return x 

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        eps = torch.randn_like(mu)
        z = eps * torch.exp(0.5 * logvar) + mu
        return z


class BaseDecoder(BaseStack):
    """
    Args:
    - latent_sz: Latent vector size
    - hidden_layers: List of hidden layer sizes
    - out_sz: Output size
    """
    def __init__(self,  latent_sz: int, hidden_layers: list, out_sz: int, use_residual: bool=True, use_drop: bool=True, dropout: float = 0.3, use_bn: bool=True):
        super().__init__(latent_sz, hidden_layers, use_residual=use_residual, use_drop=use_drop, dropout=dropout, use_bn=use_bn)
        self.out_layer = nn.Linear(hidden_layers[-1], out_sz)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = super().forward(z)
        y = self.out_layer(z)
        return y


class BaseClassifier(BaseStack):
    """
    Args:
    - latent_sz: Latent vector size
    - hidden_layers: List of hidden layer sizes
    - num_cells: Number of output classes  
    """
    def __init__(self, latent_sz:int, hidden_layers:list, num_classes:int, use_residual:bool=True, use_drop:bool=True, dropout:float=0.3, use_bn:bool=True):
        super().__init__(latent_sz, hidden_layers, use_residual=use_residual, use_drop=use_drop, dropout=dropout, use_bn=use_bn)
        self.out_layer = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        y = self.out_layer(x)
        return y
        

