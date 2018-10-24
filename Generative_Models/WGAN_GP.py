import os
import numpy as np

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch

import torch.optim as optim
from utils import *

from Generative_Models.Generative_Model import GenerativeModel

import time


class WGAN_GP(GenerativeModel):
    def __init__(self, args):

        super(WGAN_GP, self).__init__(args)


        self.model_name = 'WGAN_GP'
        self.lambda_ = 0.25

        # Loss weight for gradient penalty
        self.lambda_gp = 0.1 # 0.25 #10
        self.cuda = True
        self.c = 0.01  # clipping value
        self.n_critic = 2  # the number of iterations of the critic per generator iteration
        self.Tensor = torch.cuda.FloatTensor if True else torch.FloatTensor


        self.y_real_ = torch.FloatTensor([1])
        self.y_fake_ = self.y_real_ * -1


        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(self.device), self.y_fake_.cuda(self.device)



    def random_tensor(self, batch_size, z_dim):
        # From Normal distribution for VAE and CVAE
        return torch.randn((batch_size, z_dim, 1, 1))

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):

        self.G.train()
        self.D.train()

        epoch_start_time = time.time()
        sum_loss_train = 0.

        for iter, (x_, t_ ) in enumerate(train_loader):

            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            x_ = variable(x_)
            z_ = variable(self.random_tensor(x_.size(0), self.z_dim))

            # update D network
            self.D_optimizer.zero_grad()

            x_ = x_.view(-1, self.input_size, self.size, self.size)

            D_real = self.D(x_)
            D_real_loss = -torch.mean(D_real)

            G_ = self.G(z_)
            D_fake = self.D(G_)
            D_fake_loss = torch.mean(D_fake)

            # gradient penalty
            if self.gpu_mode:
                alpha = torch.rand((x_.size(0), 1, 1, 1)).cuda()
            else:
                alpha = torch.rand((x_.size(0), 1, 1, 1))

            x_hat = Variable(alpha * x_.data + (1 - alpha) * G_.data, requires_grad=True)

            if self.gpu_mode:
                x_hat=x_hat.cuda()

            pred_hat = self.D(x_hat.view(-1, self.input_size, self.size, self.size))

            gradients = grad(outputs=pred_hat, inputs=x_hat,
                                      grad_outputs=torch.ones(pred_hat.size()).cuda() if self.gpu_mode else torch.ones(
                                          pred_hat.size()),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
            """
            if self.gpu_mode:
                gradients = \
                    grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
            else:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = self.lambda_gp * (
                (gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
            """
            D_loss = D_real_loss + D_fake_loss + gradient_penalty

            D_loss.backward()
            self.D_optimizer.step()

            if ((iter + 1) % self.n_critic) == 0:

                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                # update G network
                self.G_optimizer.zero_grad()

                z_ = variable(self.random_tensor(x_.size(0), self.z_dim))

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = -torch.mean(D_fake)

                G_loss.backward()
                self.G_optimizer.step()

        #the following line is probably wrong
        self.train_hist['total_time'].append(time.time() - epoch_start_time)


        if self.verbose:
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                            self.epoch, self.train_hist['total_time'][0]))
            print("Training finish!... save training results")

        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        self.save()

        sum_loss_train = sum_loss_train / np.float(len(train_loader))

        return sum_loss_train




