import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from .utils import normal_init

class DenseNet(nn.Module):
    def __init__(self, input_size, layer_sizes, activation=nn.ReLU()):
        super().__init__()
        layers = []
        prev_size = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))  #
            layers.append(activation)  
            prev_size = size  
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )
        return h

class Model(nn.Module):
    def __init__(self, input_size=40, number_components=10, pseudoinputs_mean = -0.05, pseudoinputs_std=0.01):
        super(Model, self).__init__()   

        self.number_components = number_components
        self.input_size = input_size
        self.pseudoinputs_mean = pseudoinputs_mean
        self.pseudoinputs_std = pseudoinputs_std
        self.use_training_data_init = False
        self.cuda = True

    # AUXILIARY METHODS
    def add_pseudoinputs(self):

        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)

        self.means = NonLinear(self.number_components, np.prod(self.input_size), bias=False, activation=nonlinearity)

        # init pseudo-inputs
        if self.use_training_data_init:
            self.means.linear.weight.data = self.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.pseudoinputs_mean, self.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = Variable(torch.eye(self.number_components, self.number_components), requires_grad=False)
        if self.cuda:
            self.idle_input = self.idle_input.cuda()

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.