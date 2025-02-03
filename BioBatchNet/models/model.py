from .modules import *
from .VampPrior.vampprior import VampEncoder
from torch import nn
import torch.nn.functional as F

class IMCVAE(nn.Module):
    def __init__(self, **args):
        super(IMCVAE, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)
    
        if self.use_vamp:
            vamp_encoder_args = getattr(self, 'vamp_encoder_args', {})
            self.bio_encoder = VampEncoder(**vamp_encoder_args)
        else: 
            self.bio_encoder = BaseEncoder(self.in_sz, self.bio_encoder_hidden_layers, self.latent_sz)

        self.batch_encoder = BaseEncoder(self.in_sz, self.batch_encoder_hidden_layers, self.latent_sz)
        
        self.decoder = BaseDecoder(2 * self.latent_sz, self.decoder_hidden_layers, self.out_sz)
        self.bio_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_power, self.num_batch)
        self.batch_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_weak, self.num_batch)
        
        self.alpha = 1
        self.grl = GRL(alpha=self.alpha)

    def forward(self, x):  
        # bio information 
        if self.use_vamp:
            z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, bio_z = self.bio_encoder(x)
        else:
            bio_z, mu1, logvar1 = self.bio_encoder(x)

        # batch information
        batch_z, batch_mu, batch_logvar = self.batch_encoder(x)

        # combine information
        z_combine = torch.cat([bio_z, batch_z.detach()], dim=1)

        # adversarial
        bio_z_grl = self.grl(bio_z)
        bio_batch_pred = self.bio_classifier(bio_z_grl)
        batch_batch_pred = self.batch_classifier(batch_z)

        # reconstruction
        reconstruction = self.decoder(z_combine)

        if self.use_vamp:
            return bio_z, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, reconstruction
        else:
            return bio_z, mu1, logvar1, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, reconstruction
        
class GeneVAE(nn.Module):
    def __init__(self, **args):
        super(GeneVAE, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)
    
        if self.use_vamp:
            vamp_encoder_args = getattr(self, 'vamp_encoder_args', {})
            self.bio_encoder = VampEncoder(**vamp_encoder_args)
        else: 
            self.bio_encoder = BaseEncoder(self.in_sz, self.bio_encoder_hidden_layers, self.latent_sz)

        self.batch_encoder = BaseEncoder(self.in_sz, self.batch_encoder_hidden_layers, self.latent_sz)
        
        self.decoder = BaseDecoder(2*self.latent_sz, self.decoder_hidden_layers, out_sz=1000)
        self.mean_decoder = nn.Sequential(nn.Linear(1000,  self.out_sz), MeanAct())  
        self.dispersion_decoder = nn.Sequential(nn.Linear(1000,  self.out_sz), DispAct())
        self.dropout_decoder = nn.Sequential(nn.Linear(1000,  self.out_sz), nn.Sigmoid())

        self.bio_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_power, self.num_batch)
        self.batch_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_weak, self.num_batch)

        self.size_encoder = BaseEncoder(self.in_sz, self.bio_encoder_hidden_layers, 1)

        self.alpha = 1
        self.grl = GRL(alpha=self.alpha)

    def forward(self, x): 
        # bio information 
        if self.use_vamp:
            z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, bio_z = self.bio_encoder(x)
        else:
            bio_z, mu1, logvar1 = self.bio_encoder(x)
            logvar1 = torch.clamp(logvar1, min=-10, max=10)

        size_factor, size_mu, size_logvar = self.size_encoder(x)
        # batch information
        batch_z, batch_mu, batch_logvar = self.batch_encoder(x)

        # combine information\
        z_combine = torch.cat([bio_z, batch_z.detach()], dim=1)

        # adversarial
        bio_z_grl = self.grl(bio_z)
        bio_batch_pred = self.bio_classifier(bio_z_grl)
        batch_batch_pred = self.batch_classifier(batch_z)

        # h
        h = self.decoder(z_combine)
       
        # zinb output
        size_factor = torch.clamp(size_factor, min=-10, max=10)
        _mean = self.mean_decoder(h) * torch.exp(size_factor)
        _disp = self.dispersion_decoder(h)
        _pi = self.dropout_decoder(h)    

        if self.use_vamp:
            return bio_z, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, _mean, _disp, _pi
        else:
            return bio_z, mu1, logvar1, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, _mean, _disp, _pi, size_factor, size_mu, size_logvar
        
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-3, max=1e3)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-3, max=1e3)