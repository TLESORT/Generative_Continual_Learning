from Training.Trainer import Trainer
from torch.nn import functional as F
from utils import variable
import torch.nn as nn
import torch
import random
import numpy as np
from torch.autograd import Variable
from copy import deepcopy


class Ewc(Trainer):
    def __init__(self, model, args):
        super(Ewc, self).__init__(model, args)
        self.params = None
        self._means = None
        self._precision_matrices = None
        self.importance = args.lambda_EWC

    def penalty(self, model: nn.Module):

        if self.context == 'Classification':
            models = [model]
        elif self.context == 'Generation':
            if model.model_name in ['CVAE', 'VAE']:
                models = [model.G, model.E]
            elif model.model_name in ['CGAN', 'GAN', 'WGAN', 'BEGAN']:
                models = [model.G]

        loss = 0
        for model_ in models:
            for n, p in model_.named_parameters():
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

    def _diag_fisher(self, model):

        if self.context == 'Classification':

            precision_matrices = {}
            for n, p in deepcopy(self.params).items():
                p.data.zero_()
                precision_matrices[n] = variable(p.data)

            model.eval()

            for input in self.old_task:

                model.zero_grad()
                input = variable(input).view(-1, 1)
                output = model(input).view(1, -1)
                label = output.max(1)[1].view(-1)
                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()

                for n, p in model.named_parameters():
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.old_task)

            precision_matrices = {n: p for n, p in precision_matrices.items()}
            return precision_matrices

        if self.context == 'Generation':

            old_batch_size = self.train_loader.batch_size
            self.train_loader.batch_size = 1
            #model.batch_size = 1

            precision_matrices = {}
            for n, p in deepcopy(self.params).items():
                p.data.zero_()
                precision_matrices[n] = variable(p.data)

            model.eval()

            self.y_real_ = variable(torch.ones(1, 1))
            self.y_fake_ = variable(torch.zeros(1, 1))

            if model.model_name in ['CVAE', 'VAE']:

                models = [model.G, model.E]

                model.E.eval()
                model.G.eval()

                for iter, (x_, t_) in enumerate(self.train_loader):

                    self.model.E_optimizer.zero_grad()
                    self.model.G_optimizer.zero_grad()

                    x_ = variable(x_)

                    if model.model_name == 'CVAE':
                        y_onehot = variable(model.get_one_hot(t_))
                        z_, mu, logvar = model.E(x_, y_onehot)
                        recon_batch = model.G(z_, y_onehot)
                    else:
                        z_, mu, logvar = model.E(x_)
                        recon_batch = model.G(z_)

                    loss = model.loss_function(recon_batch, x_, mu, logvar)
                    loss = torch.log(loss)
                    loss.backward()

                    for model_ in models:
                        for n, p in model_.named_parameters():
                            precision_matrices[n].data += p.grad.data ** 2 / int(len(self.train_loader))

            elif model.model_name in ["GAN", "CGAN" , "WGAN"]:

                models = [model.G]

                self.model.G.eval()
                self.model.D.eval()

                for iter, (x_, t_) in enumerate(self.train_loader):

                    self.model.G_optimizer.zero_grad()
                    self.model.D_optimizer.zero_grad()


                    z_ = variable(torch.rand((1, self.model.z_dim)))

                    if model.model_name == 'CGAN':
                        y_onehot = variable(model.get_one_hot(t_))
                        G_ = self.model.G(z_, y_onehot)
                        D_fake = self.model.D(G_, y_onehot)
                        BCELoss = nn.BCELoss()
                        G_loss = BCELoss(D_fake[0], self.y_real_)
                        G_loss = torch.log(G_loss)
                        G_loss = torch.mean(G_loss)

                    elif model.model_name == 'GAN':

                        G_ = model.G(z_)
                        D_fake = model.D(G_)
                        G_loss = model.BCELoss(D_fake, self.y_real_)
                        G_loss = torch.log(G_loss)

                    elif model.model_name == 'WGAN':

                        z_ = variable(torch.rand((1, model.z_dim, 1, 1)))

                        # clipping D
                        for p in model.D.parameters():
                            p.data.clamp_(-model.c, model.c)

                        G_ = model.G(z_)
                        D_fake = model.D(G_)
                        D_fake = torch.log(D_fake)
                        G_loss = -torch.mean(D_fake)

                    G_loss.backward()

                    for model_ in models:
                        for n, p in model_.named_parameters():
                            precision_matrices[n].data += p.grad.data ** 2 / int(len(self.train_loader))


            '''if model.model_name == 'BEGAN':

                precision_matrices = {n: p for n, p in precision_matrices.items()}
                return precision_matrices'''

        else:
            print('Not implemented yet')
            return None

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.train_loader.batch_size = old_batch_size
        model.batch_size = old_batch_size
        return precision_matrices

    def additional_loss(self, model):
        if self.ind_task > 0:
            loss = self.importance * self.penalty(model)
        else:
            loss = None
        return loss

    def preparation_4_task(self, model, ind_task):

        if self.context == 'Classification':

            # Here model is only the generator.

            train_loader, test_loader = self.create_next_data(ind_task)

            self.model.train()

            # we only save the importance of weights at the beginning of each new task
            old_tasks = []
            for sub_task in range(self.ind_task + 1):
                old_tasks = old_tasks + list(self.train_loader[sub_task].get_sample(self.samples_per_task))
            old_tasks = random.sample(old_tasks, k=self.samples_per_task)
            self.old_task = old_tasks

            self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            self._means = {}
            self._precision_matrices = self._diag_fisher(model)

            for n, p in deepcopy(self.params).items():
                self._means[n] = variable(p.data)

            return train_loader, test_loader

        if ind_task > 0 and self.context == 'Generation':

            ### Compute Fischer info matrix
            self.params = {n: p for n, p in model.G.named_parameters() if p.requires_grad}

            if model.model_name in ['CVAE', 'VAE']:
                self.params_E = {n: p for n, p in model.E.named_parameters() if p.requires_grad}
                self.params = {**self.params, **self.params_E}

            self._means = {}
            precision_matrices_this_task = self._diag_fisher(model)

            # update fisher info with previous tasks fisher info, and this task fisher info (sum)
            if ind_task == 1:
                self._precision_matrices = precision_matrices_this_task
            else:
                self._precision_matrices = {n: p + self._precision_matrices[n] for n, p in
                                            precision_matrices_this_task.items()}

            for n, p in deepcopy(self.params).items():
                self._means[n] = variable(p.data)

            ### 
            nb_sample_train = len(self.train_loader[ind_task])
            nb_sample_test = int(nb_sample_train * 0.2)

            # we generate dataset for later evaluation
            nb_sample_train = self.sample_transfer #len(self.train_loader[ind_task])
            nb_sample_test = int(nb_sample_train * 0.2)
            self.model.generate_dataset(ind_task - 1, nb_sample_train, one_task=False, Train=True)
            #self.model.generate_dataset(ind_task - 1, nb_sample_test, one_task=False, Train=False)


            train_loader, test_loader = self.create_next_data(ind_task)

            return train_loader, test_loader

        train_loader, test_loader = self.create_next_data(ind_task)
        return train_loader, test_loader
