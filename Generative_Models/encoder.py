
import torch
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, z_dim, dataset='mnist', conditional=False):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        if dataset == 'mnist' or dataset == 'fashion':
            self.input_size = 784
        elif dataset == 'celebA':
            self.input_size = 64 * 64 * 3
        elif dataset == 'cifar10':
            self.input_size = 32 * 32 * 3
            # self.input_size = 64 * 64 * 3
        elif dataset == 'timagenet':
            self.input_size = 64 * 64 * 3
        if self.conditional:
            self.input_size += 10
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(self.input_size, 1200)
        self.fc21 = nn.Linear(1200, z_dim)
        self.fc22 = nn.Linear(1200, z_dim)

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat([x, c], 1)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_() # does not work for other device than 0
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, c=None):
        mu, logvar = self.encode(x.view(x.size(0), -1), c)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar