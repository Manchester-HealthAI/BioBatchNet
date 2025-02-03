from .nn import NonLinear, Model, DenseNet
from .utils import *


import torch.nn as nn
import torch
import math
import numpy as np


"""
    - args.input_size
    - args.z1_size
    - args.z2_size
    - args.prior
    - args.number_components
"""
class VampVAE(Model):
    def __init__(self, args) -> None:
        self.args = args

        # encoder q(z2 | x)
        self.q_z2_layers = DenseNet(input_size = self.args.input_size, layer_sizes = [64, 128])

        self.q_z2_mean = nn.Linear(128, self.args.z2_size) 
        self.q_z2_logvar = NonLinear(128, self.args.z2_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # encoder q(z1 | z2, x)
        self.q_z1_layers_x = DenseNet(np.prod(self.args.input_size), [64, 128])
        self.q_z1_layers_z2 = DenseNet(self.args.z2_size, [64, 128])
        self.q_z1_layers_joint = DenseNet(2 * 128, 128)

        self.q_z1_mean = nn.Linear(128, self.args.z1_size) 
        self.q_z1_logvar = NonLinear(128, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(z1 | z2)
        self.p_z1_layers = DenseNet(self.args.z2_size, [64, 128])  

        self.p_z1_mean = nn.Linear(128, self.args.z1_size)  
        self.p_z1_logvar = NonLinear(128, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z1, z2)
        self.p_x_layers_z1 = DenseNet(self.args.z1_size, [64, 128])  # extract information from z1
        self.p_x_layers_z2 = DenseNet(self.args.z2_size, [64, 128])  # extract information from z2
        self.p_x_layers_joint = DenseNet(2 * 128, [128, 256])  # joint z1 and z2

        self.p_x_mean = NonLinear(256, np.prod(self.args.input_size), activation=nn.Sigmoid())
        self.p_x_logvar = NonLinear(256, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))

        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()
    
    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z2(self, x):
        # q(z2 | x)
        x = self.q_z2_layers(x)

        z2_q_mean = self.q_z2_mean(x)
        z2_q_logvar = self.q_z2_logvar(x)
        return z2_q_mean, z2_q_logvar

    def q_z1(self, x, z2):
        # q(z1 | x, z2)
        x = self.q_z1_layers_x(x)

        z2 = self.q_z1_layers_z2(z2)

        h = torch.cat((x, z2), 1)
        h = self.q_z1_layers_joint(h)

        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_z1(self, z2):
        # p(z1 | z2)
        z2 = self.p_z1_layers(z2)

        z1_mean = self.p_z1_mean(z2)
        z1_logvar = self.p_z1_logvar(z2)
        return z1_mean, z1_logvar

    def p_x(self, z1, z2):
        # p(x | z1, z2)
        z1 = self.p_x_layers_z1(z1)

        z2 = self.p_x_layers_z2(z2)

        h = torch.cat((z1, z2), 1)
        h = self.p_x_layers_joint(h)

        x_mean = self.p_x_mean(h)
        x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
        x_logvar = self.p_x_logvar(h)
        return x_mean, x_logvar

    # Prior
    def log_p_z2(self, z2):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z2, dim=1)

        elif self.args.prior == 'vampprior':
            # z2 - MB x M
            C = self.args.number_components

            # calculate params
            X = self.means(self.idle_input)

            # calculate params for given data
            z2_p_mean, z2_p_logvar = self.q_z2(X)  # C x M

            # expand z
            z_expand = z2.unsqueeze(1)
            means = z2_p_mean.unsqueeze(0)
            logvars = z2_p_logvar.unsqueeze(0)

            a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
            a_max, _ = torch.max(a, 1)  # MB

            # calculte log-sum-exp
            log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior
    
    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # z2 ~ q(z2 | x)
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # z1 ~ q(z1 | x, z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # p(z1 | z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # x_mean = p(x|z1,z2)
        x_mean, x_logvar = self.p_x(z1_q, z2_q)

        return x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar
    
    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)  
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_p_z2 = self.log_p_z2(z2_q)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)
        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

        loss = -RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL


class VampEncoder(Model):
    def __init__(self, input_size=2000, z1_size=20, z2_size=20, number_components=10):
        super(VampEncoder, self).__init__()
        self.number_components = number_components

        # encoder q(z2 | x)
        self.q_z2_layers = DenseNet(input_size=input_size, layer_sizes=[64, 128])
        self.q_z2_mean = nn.Linear(128, z2_size)
        self.q_z2_logvar = NonLinear(128, z2_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # encoder q(z1 | z2, x)
        self.q_z1_layers_x = DenseNet(np.prod(input_size), [64, 128])
        self.q_z1_layers_z2 = DenseNet(z2_size, [64, 128])
        self.q_z1_layers_joint = DenseNet(2 * 128, [128])

        self.q_z1_mean = nn.Linear(128, z1_size)
        self.q_z1_logvar = NonLinear(128, z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))
    
        # decoder: p(z1 | z2)
        self.p_z1_layers = DenseNet(z2_size, [64, 128])  

        self.p_z1_mean = nn.Linear(128, z1_size)  
        self.p_z1_logvar = NonLinear(128, z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder p(z1, z2)
        self.p_x_layers_z1 = DenseNet(z1_size, [64, 20])  # extract information from z1
        self.p_x_layers_z2 = DenseNet(z2_size, [64, 20])  # extract information from z2
        self.p_x_layers_joint = DenseNet(2 * 20, [20])  # joint z1 and z2

        self.add_pseudoinputs()

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z2(self, x):
        # q(z2 | x)
        print(self.q_z2_layers)
        x = self.q_z2_layers(x)

        z2_q_mean = self.q_z2_mean(x)
        z2_q_logvar = self.q_z2_logvar(x)
        return z2_q_mean, z2_q_logvar

    def q_z1(self, x, z2):
        # q(z1 | x, z2)
        x = self.q_z1_layers_x(x)

        z2 = self.q_z1_layers_z2(z2)

        h = torch.cat((x, z2), 1)
        h = self.q_z1_layers_joint(h)

        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_z1(self, z2):
        # p(z1 | z2)
        z2 = self.p_z1_layers(z2)

        z1_mean = self.p_z1_mean(z2)
        z1_logvar = self.p_z1_logvar(z2)
        return z1_mean, z1_logvar

    def p_z1z2(self, z1, z2):
        # p(x | z1, z2)
        z1 = self.p_x_layers_z1(z1)

        z2 = self.p_x_layers_z2(z2)

        h = torch.cat((z1, z2), 1)
        z_joint = self.p_x_layers_joint(h)

        return z_joint

    # Prior
    def log_p_z2(self, z2):

        # z2 - MB x M
        C = self.number_components

        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        z2_p_mean, z2_p_logvar = self.q_z2(X)  # C x M

        # expand z
        z_expand = z2.unsqueeze(1)
        means = z2_p_mean.unsqueeze(0)
        logvars = z2_p_logvar.unsqueeze(0)

        # GMM
        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB

        # calculte log-sum-exp
        log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        return log_prior


    # THE MODEL: ENCODER
    def forward(self, x):
        # z2 ~ q(z2 | x)
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # z1 ~ q(z1 | x, z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # p(z1 | z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # p(z1,z2)
        z_joint = self.p_z1z2(z1_q, z2_q)

        return z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, z_joint
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # KL Loss function
    def Vamp_KL_loss(self, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar, beta=1., average=True):

        # KL Divergence
        log_p_z1 = log_Normal_diag(z1_q, z1_p_mean, z1_p_logvar, dim=1)
        log_q_z1 = log_Normal_diag(z1_q, z1_q_mean, z1_q_logvar, dim=1)
        log_p_z2 = self.log_p_z2(z2_q)
        log_q_z2 = log_Normal_diag(z2_q, z2_q_mean, z2_q_logvar, dim=1)

        KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
        Vamp_KL_loss = beta * KL

        if average:
            Vamp_KL_loss = torch.mean(Vamp_KL_loss)

        return Vamp_KL_loss


