import torch
import torch.nn as nn


def Generator(z_dim=62, dataset='mnist', conditional=False, model='VAE'):
    if dataset == 'mnist' or dataset == 'fashion':
        return MNIST_Generator(z_dim, dataset, conditional, model)
        # else:
        #    raise ValueError("This generator is not implemented")


class MNIST_Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim=62, dataset='mnist', conditional=False, model='VAE'):
        super(MNIST_Generator, self).__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.model = model
        self.conditional = conditional

        self.latent_dim = 1024

        self.input_height = 28
        self.input_width = 28
        self.input_dim = z_dim
        if self.conditional:
            self.input_dim += 10
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.maxPool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.Sigmoid = nn.Sigmoid()
        self.apply(self.weights_init)

    def forward(self, input, c=None):

        if c is not None:
            input = input.view(-1, self.input_dim - 10)
            input = torch.cat([input, c], 1)
        else:
            input = input.view(-1, self.input_dim)

        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)
        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Generator_Cifar(nn.Module):
    def __init__(self, z_dim, conditional=False):
        super(Generator_Cifar, self).__init__()
        self.nc = 3

        self.conditional = conditional

        self.nz = z_dim
        if self.conditional:
            self.nz += 10

        self.ngf = 64
        self.ndf = 64
        self.ngpu = 1

        self.Conv1 = nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False)
        self.BN1 = nn.BatchNorm2d(self.ngf * 8)
        self.Relu = nn.ReLU(True)
        # state size. (ngf*8) x 4 x 4
        self.Conv2 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(self.ngf * 4)
        # nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        self.Conv3 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(self.ngf * 2)
        # nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        # nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
        self.Conv4 = nn.ConvTranspose2d(self.ngf * 2, self.nc, 4, 2, 1, bias=False)
        # nn.BatchNorm2d(self.ngf),
        # nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        # nn.ConvTranspose2d(self.ngf,self.nc, 4, 2, 1, bias=False),
        self.Tanh = nn.Tanh()
        # state size. (nc) x 64 x 64

        '''
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            #nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(self.ngf * 2,self.nc, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ngf),
            #nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(self.ngf,self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        '''

    def forward(self, input, c=None):

        if c is not None:
            input = input.view(-1, self.nz - 10, 1, 1)
            input = torch.cat([input, c], 1)
        else:
            input = input.view(-1, self.nz, 1, 1)

        x = self.Relu(self.BN1(self.Conv1(input)))
        x = self.Relu(self.BN2(self.Conv2(x)))
        x = self.Relu(self.BN3(self.Conv3(x)))
        x = self.Tanh(self.Conv4(x))

        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
