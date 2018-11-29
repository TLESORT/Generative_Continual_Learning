import utils, torch, time
import numpy as np
from torch.autograd import Variable
from Generative_Models.Conditional_Model import ConditionalModel
from Data.load_dataset import get_iter_dataset
from torch.utils.data import DataLoader

from utils import variable

import math


class CGAN(ConditionalModel):


    def run_batch(self, x_, t_, additional_loss=None):

        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        self.G.train()
        self.D.train()

        x_ = x_.view((-1, 1, 28, 28))
        y_onehot = variable(self.get_one_hot(t_))
        z_ = variable(torch.rand((x_.size(0), self.z_dim)))

        # update D network
        self.D_optimizer.zero_grad()

        D_real = self.D(x_, y_onehot)

        D_real_loss = self.BCELoss(D_real[0], self.y_real_[:x_.size(0)])

        G_ = self.G(z_, y_onehot)

        D_fake = self.D(G_, y_onehot)
        D_fake_loss = self.BCELoss(D_fake[0], self.y_fake_[:x_.size(0)])

        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        self.D_optimizer.step()

        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update

        # update G network
        self.G_optimizer.zero_grad()

        G_ = self.G(z_, y_onehot)
        D_fake = self.D(G_, y_onehot)
        G_loss = self.BCELoss(D_fake[0], self.y_real_[:x_.size(0)])


        if additional_loss is not None:
            regularization = additional_loss(self)
            G_loss += regularization

        G_loss.backward()
        self.G_optimizer.step()
        return G_loss.item()

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):

        self.G.train()
        self.D.train()
        epoch_start_time = time.time()

        sum_loss_train = 0.


        for iter, (x_, t_) in enumerate(train_loader):


            if self.num_task==10 and ind_task != self.num_task :

                # An image can be wrongly labelled by a label from futur task
                # it is not a ethical problem and it help learning

                # if ind_task != self.num_task, there no more future task, nothing to do

                # the following line produce a vector of batch_size with label from ind_task to self.num_task-1
                rand_t_ = ((torch.randperm(1000) % (self.num_task-ind_task)).long() + ind_task)[:x_.size(0)]
                mask = (t_ == ind_task).long()

                # if we are in a past task we keep the true label
                # else we put a random label from the futur label
                t_ = torch.mul(t_, 1 - mask) + torch.mul(rand_t_, mask)

            x_ = variable(x_.view((-1, 1, 28, 28)))
            sum_loss_train+=self.run_batch(x_, t_, additional_loss=None)


        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        self.save()

        sum_loss_train = sum_loss_train / np.float(len(train_loader))

        return sum_loss_train