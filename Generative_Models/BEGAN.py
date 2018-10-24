import utils, torch, time, os
import numpy as np
from torch.autograd import Variable
from Generative_Models.Generative_Model import GenerativeModel
from Data.load_dataset import get_iter_dataset




class BEGAN(GenerativeModel):
    def __init__(self, args):
        super(BEGAN, self).__init__(args)
        self.gamma = 0.75
        self.lambda_ = 0.001
        self.k = 0.

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):
        self.size_epoch = 1000

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda(self.device)), Variable(
                torch.zeros(self.batch_size, 1).cuda(self.device))
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(
                torch.zeros(self.batch_size, 1))

        self.G.train()
        self.D.train()

        epoch_start_time = time.time()
        sum_loss_train = 0.

        for iter, (x_, t_) in enumerate(train_loader):

            if x_.size(0) != self.batch_size:
                break

            x_ = x_.view((-1, 1, 28, 28))
            z_ = torch.rand((self.batch_size, self.z_dim))

            if self.gpu_mode:
                x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
            else:
                x_, z_ = Variable(x_), Variable(z_)

            # update D network
            self.D_optimizer.zero_grad()

            D_real = self.D(x_)
            D_real_err = torch.mean(torch.abs(D_real - x_))

            G_ = self.G(z_)
            D_fake = self.D(G_)
            D_fake_err = torch.mean(torch.abs(D_fake - G_))

            D_loss = D_real_err - self.k * D_fake_err
            self.train_hist['D_loss'].append(D_loss.data[0])

            D_loss.backward()
            self.D_optimizer.step()

            # update G network
            self.G_optimizer.zero_grad()

            G_ = self.G(z_)
            D_fake = self.D(G_)
            D_fake_err = torch.mean(torch.abs(D_fake - G_))

            G_loss = D_fake_err
            self.train_hist['G_loss'].append(G_loss.data[0])

            G_loss.backward()
            self.G_optimizer.step()

            # convergence metric
            temp_M = D_real_err + torch.abs(self.gamma * D_real_err - D_fake_err)

            # operation for updating k
            temp_k = self.k + self.lambda_ * (self.gamma * D_real_err - D_fake_err)
            temp_k = temp_k.data[0]

            # self.k = temp_k.data[0]
            self.k = min(max(temp_k, 0), 1)
            self.M = temp_M.data[0]

            if self.verbose:
                if ((iter + 1) % 100) == 0:
                    print("Ind_task : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f" %
                          (ind_task, (epoch + 1), (iter + 1), self.size_epoch,
                           D_loss.data[0], G_loss.data[0], self.M, self.k))

        # the following line is probably wrong
        self.train_hist['total_time'].append(time.time() - epoch_start_time)


        if self.verbose:
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                            self.epoch,
                                                                            self.train_hist['total_time'][0]))
            print("Training finish!... save training results")

        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        self.save()

        sum_loss_train = sum_loss_train / np.float(len(train_loader))

        return sum_loss_train

    '''
    def train(self):

        self.G.apply(self.G.weights_init)
        self.D.train()

        for classe in range(10):
            self.train_hist = {}
            self.train_hist['D_loss'] = []
            self.train_hist['G_loss'] = []
            self.train_hist['per_epoch_time'] = []
            self.train_hist['total_time'] = []
            # self.G.apply(self.G.weights_init) does not work for instance

            if self.gpu_mode:
                self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
            else:
                self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

            self.D.train()
            self.data_loader_train = get_iter_dataset(self.dataset_train, self.list_class_train, self.batch_size,
                                                      classe)
            self.data_loader_valid = get_iter_dataset(self.dataset_valid, self.list_class_valid, self.batch_size,
                                                      classe)
            print('training class : ' + str(classe))
            start_time = time.time()
            for epoch in range(self.epoch):
                self.G.train()
                epoch_start_time = time.time()
                n_batch = 0.

                for iter, (x_, t_) in enumerate(self.data_loader_train):
                    n_batch += 1
                    z_ = torch.rand((self.batch_size, self.z_dim))

                    if self.gpu_mode:
                        x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                    else:
                        x_, z_ = Variable(x_), Variable(z_)

                    # update D network
                    self.D_optimizer.zero_grad()

                    D_real = self.D(x_)
                    D_real_err = torch.mean(torch.abs(D_real - x_))

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_err = torch.mean(torch.abs(D_fake - G_))

                    D_loss = D_real_err - self.k * D_fake_err
                    self.train_hist['D_loss'].append(D_loss.data[0])

                    D_loss.backward()
                    self.D_optimizer.step()

                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_err = torch.mean(torch.abs(D_fake - G_))

                    G_loss = D_fake_err
                    self.train_hist['G_loss'].append(G_loss.data[0])

                    G_loss.backward()
                    self.G_optimizer.step()

                    # convergence metric
                    temp_M = D_real_err + torch.abs(self.gamma * D_real_err - D_fake_err)

                    # operation for updating k
                    temp_k = self.k + self.lambda_ * (self.gamma * D_real_err - D_fake_err)
                    temp_k = temp_k.data[0]

                    # self.k = temp_k.data[0]
                    self.k = min(max(temp_k, 0), 1)
                    self.M = temp_M.data[0]

                    if ((iter + 1) % 100) == 0:
                        print("classe : [%1d] Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, M: %.8f, k: %.8f" %
                              (classe, (epoch + 1), (iter + 1), self.size_epoch,
                               D_loss.data[0], G_loss.data[0], self.M, self.k))

                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                self.visualize_results((epoch+1), classe)

            self.save_G(classe)

            result_dir = self.result_dir + '/' + 'classe-' + str(classe)
            utils.generate_animation(result_dir + '/' + self.model_name, epoch + 1)
            utils.loss_plot(self.train_hist, result_dir, self.model_name)

            np.savetxt(
                os.path.join(result_dir, 'began_training_' + self.dataset + '.txt'),
                np.transpose([self.train_hist['G_loss']]))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    '''