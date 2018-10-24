import utils, torch, time
import numpy as np
from torch.autograd import Variable
from Generative_Models.Generative_Model import GenerativeModel
from Data.load_dataset import get_iter_dataset
from torch.utils.data import DataLoader

from utils import variable

import math


class CGAN(GenerativeModel):

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):

        self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))

        self.G.train()
        self.D.train()
        epoch_start_time = time.time()

        sum_loss_train = 0.

        count = np.zeros(10)

        for iter, (x_, t_) in enumerate(train_loader):

            # this case happen for the last batch of the train_loader
            if x_.size(0) != self.batch_size:
                self.y_real_, self.y_fake_ = variable(torch.ones(x_.size(0), 1).cuda(self.device)), variable(
                    torch.zeros(x_.size(0), 1).cuda(self.device))
                # This is the last iteration we don't need to set back the shape to self.batch_size

            if x_.size(0) == 1:
                break

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
            y_onehot = variable(self.get_one_hot(t_))
            z_ = variable(torch.rand((x_.size(0), self.z_dim)))


            # update D network
            self.D_optimizer.zero_grad()

            D_real = self.D(x_, y_onehot)
            D_real_loss = self.BCELoss(D_real[0], self.y_real_)

            G_ = self.G(z_, y_onehot)

            D_fake = self.D(G_, y_onehot)
            D_fake_loss = self.BCELoss(D_fake[0], self.y_fake_)

            D_loss = D_real_loss + D_fake_loss
            self.train_hist['D_loss'].append(D_loss.item())

            D_loss.backward()
            self.D_optimizer.step()

            # update G network
            self.G_optimizer.zero_grad()

            G_ = self.G(z_, y_onehot)
            D_fake = self.D(G_, y_onehot)
            G_loss = self.BCELoss(D_fake[0], self.y_real_)
            self.train_hist['G_loss'].append(G_loss.item())
            sum_loss_train += G_loss.item()

            regularization = additional_loss(self)

            if regularization is not None:
                #print('BEFORE', G_loss)
                G_loss += regularization
                #print('AFTER', G_loss)

            G_loss.backward()
            self.G_optimizer.step()

            if self.verbose:
                if ((iter + 1) % 100) == 0:
                    print("ind_task : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          (ind_task, (epoch + 1), (iter + 1), int(len(train_loader)/self.batch_size), D_loss.item(), G_loss.item()))



        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        self.save()

        #print('COUNT IS', count)

        sum_loss_train = sum_loss_train / np.float(len(train_loader))

        return sum_loss_train