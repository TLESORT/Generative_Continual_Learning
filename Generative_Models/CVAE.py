from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

from Generative_Models.VAE import VAE

from utils import variable


class CVAE(VAE):
    def __init__(self, args):
        super(CVAE, self).__init__(args)

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):

        self.E.train()
        self.G.train()
        sum_loss_train = 0.
        n_batch = 0.

        for iter, (x_, t_) in enumerate(train_loader):

            x_ = variable(x_)
            y_onehot = variable(self.get_one_hot(t_))

            self.E_optimizer.zero_grad()
            self.G_optimizer.zero_grad()
            # VAE
            z_, mu, logvar = self.E(x_, y_onehot)
            recon_batch = self.G(z_, y_onehot) 

            G_loss = self.loss_function(recon_batch, x_.view(-1,1,28,28), mu, logvar)
            sum_loss_train += G_loss.item()

            #regularization = additional_loss([self.G, self.E])
            regularization = additional_loss(self)
            #regularization = additional_loss(self.E)

            if regularization is not None:
                G_loss += regularization

            G_loss.backward()
            self.E_optimizer.step()
            self.G_optimizer.step()

            n_batch += 1

            if self.verbose:
                if ((iter + 1) % 100) == 0:
                    print("Task : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, E_loss: %.8f" %
                          (ind_task, (epoch + 1), (iter + 1), self.size_epoch, G_loss.item(), G_loss.item()))

        sum_loss_train = sum_loss_train / np.float(n_batch)
        return sum_loss_train
