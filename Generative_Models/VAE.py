from __future__ import print_function

import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Generative_Models.Generative_Model import GenerativeModel
from Generative_Models.discriminator import Discriminator
from Generative_Models.encoder import Encoder
from Generative_Models.generator import Generator

from utils import variable


class VAE(GenerativeModel):
    def __init__(self, args):

        super(VAE, self).__init__(args)


        self.E = Encoder(self.z_dim, self.dataset, self.conditional)
        self.E_optimizer = optim.Adam(self.E.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.lr = args.lrD

        if self.gpu_mode:
            self.E.cuda(self.device)

        self.sample_z_ = variable(torch.randn((self.sample_num, self.z_dim, 1, 1)))


    def load_G(self, ind_task):
        self.G.load_state_dict(
            torch.load(os.path.join(self.save_dir, self.model_name + '-' + str(ind_task) + '_G.pkl')))

    # self.E.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + '_E.pkl')))

    # save a generator, encoder and discriminator in a given class
    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.G.state_dict(), os.path.join(self.save_dir, self.model_name + '_G.pkl'))
        torch.save(self.E.state_dict(), os.path.join(self.save_dir, self.model_name + '_E.pkl'))

        with open(os.path.join(self.save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def random_tensor(self, batch_size, z_dim):
        # From Normal distribution for VAE and CVAE
        return torch.randn((batch_size, z_dim, 1, 1))

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):

        start_time = time.time()

        n_batch = 0

        self.E.train()
        self.G.train()
        sum_loss_train = 0.

        for iter, (x_, _) in enumerate(train_loader):


            n_batch += 1
            x_ = Variable(x_)
            if self.gpu_mode:
                x_ = x_.cuda(self.device)
            # VAE
            z_, mu, logvar = self.E(x_)
            recon_batch = self.G(z_)
            # train
            self.G_optimizer.zero_grad()
            self.E_optimizer.zero_grad()
            g_loss = self.loss_function(recon_batch, x_.view(-1,1,28,28), mu, logvar)
            sum_loss_train += g_loss.item()

            #regularization = additional_loss([self.G, self.E])
            regularization = additional_loss(self)
            #regularization = additional_loss(self.E)

            if regularization is not None:
                g_loss += regularization

            g_loss.backward()
            self.G_optimizer.step()
            self.E_optimizer.step()

            if self.verbose:
                if ((iter + 1) % 100) == 0:
                    print("Task : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, E_loss: %.8f" %
                          (ind_task, (epoch + 1), (iter + 1), len(train_loader), g_loss.item(), g_loss.item()))

        sum_loss_train = sum_loss_train / np.float(n_batch)

        return sum_loss_train

    def loss_function(self, recon_x, x, mu, logvar):
        if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            reconstruction_function = nn.BCELoss()
        else:
            reconstruction_function = nn.MSELoss()
        reconstruction_function.size_average = False

        recon_x=recon_x.view(-1,self.input_size, self.size, self.size)
        x=x.view(-1,self.input_size, self.size, self.size)

        BCE = reconstruction_function(recon_x, x)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        if self.gpu_mode:
            BCE = BCE.cuda(self.device)
            KLD = KLD.cuda(self.device)
        return BCE + KLD

    def train(self):
        self.G.train()
        self.E.train()

    def eval(self):
        self.G.eval()
        self.E.eval()
