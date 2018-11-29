import numpy as np
import time
import torch
from torch.autograd import Variable
from utils import variable

from Generative_Models.Generative_Model import GenerativeModel


class GAN(GenerativeModel):


    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):
        self.size_epoch = 1000

        self.G.train()
        self.D.train()

        epoch_start_time = time.time()
        sum_loss_train = 0.


        for iter, (x_, t_ ) in enumerate(train_loader):


            if x_.size(0) != self.batch_size:
                break

            x_ = x_.view((-1, self.input_size, self.size, self.size))

            z_ = torch.rand((x_.size(0), self.z_dim))

            x_, z_ = variable(x_), variable(z_)

            # update D network
            self.D_optimizer.zero_grad()

            D_real = self.D(x_)
            D_real_loss = self.BCELoss(D_real, self.y_real_[:x_.size(0)])

            G_ = self.G(z_)
            D_fake = self.D(G_)
            D_fake_loss = self.BCELoss(D_fake, self.y_fake_[:x_.size(0)])

            D_loss = D_real_loss + D_fake_loss
            self.train_hist['D_loss'].append(D_loss.item())

            D_loss.backward()
            self.D_optimizer.step()

            # update G network
            self.G_optimizer.zero_grad()

            G_ = self.G(z_)
            D_fake = self.D(G_)
            G_loss = self.BCELoss(D_fake, self.y_real_[:x_.size(0)])
            self.train_hist['G_loss'].append(G_loss.item())
            sum_loss_train += G_loss.item()

            regularization = additional_loss(self)

            if regularization is not None:
                G_loss += regularization

            G_loss.backward()
            self.G_optimizer.step()

            if self.verbose:
                if ((iter + 1) % 100) == 0:
                    print("classe : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          (ind_task, (epoch + 1), (iter + 1), len(train_loader), D_loss.data[0], G_loss.data[0]))


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

