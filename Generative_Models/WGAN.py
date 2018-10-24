import numpy as np
import time
import torch
from utils import variable
from Generative_Models.Generative_Model import GenerativeModel


class WGAN(GenerativeModel):
    def __init__(self, args):
        super(WGAN, self).__init__(args)
        self.c = 0.01  # clipping value
        self.n_critic = 2  # the number of iterations of the critic per generator iteration

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):


        #self.y_real_ = variable(torch.ones(self.batch_size, 1))
        #self.y_fake = variable(torch.zeros(self.batch_size, 1))

        self.G.train()
        self.D.train()

        epoch_start_time = time.time()
        sum_loss_train = 0.


        for iter, (x_, t_ ) in enumerate(train_loader):

            x_ = variable(x_.view((-1, self.input_size, self.size, self.size)))
            z_ = variable(torch.rand((self.batch_size, self.z_dim, 1, 1)))


            self.D_optimizer.zero_grad()
            D_real = self.D(x_)
            D_real_loss = -torch.mean(D_real)

            G_ = self.G(z_)
            D_fake = self.D(G_)
            D_fake_loss = torch.mean(D_fake)

            D_loss = D_real_loss + D_fake_loss

            D_loss.backward()
            self.D_optimizer.step()

            # clipping D
            for p in self.D.parameters():
                p.data.clamp_(-self.c, self.c)

            if ((iter + 1) % self.n_critic) == 0:
                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = -torch.mean(D_fake)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                self.train_hist['D_loss'].append(D_loss.item())

            if self.verbose:
                if ((iter + 1) % 100) == 0:
                    print("ind_task : [%1d] Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f" %
                          (ind_task, (epoch + 1), (iter + 1), self.size_epoch, G_loss.item(), D_loss.item()))


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

