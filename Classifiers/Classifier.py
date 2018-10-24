import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import variable
import torch.optim as optim
from utils import variable
import numpy as np
import math


class Classifier(object):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.args = args
        self.batchsize = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.device = args.device
        self.save_dir = args.save_dir
        self.verbose = args.verbose

        self.net = Net()
        if self.gpu_mode:
            self.net.cuda(self.device)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.args.lrC)

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss):
        self.net.train()
        epoch_loss = 0
        correct = 0
        train_loader.shuffle_task()
        for data, target in train_loader:
            data, target = variable(data), variable(target)

            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)

            self.optimizer.zero_grad()

            output = self.net(data)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss.item()

            if additional_loss is not None:
                regularization = additional_loss(self.net)

                if regularization is not None:
                    loss += regularization

            loss.backward()
            self.optimizer.step()
            correct += (output.max(dim=1)[1] == target).data.sum()

        if self.verbose:
            print('Train eval : task : ' + str(ind_task) + " - correct : " + str(correct) + ' / ' + str(
                len(train_loader)))

        return epoch_loss / np.float(len(train_loader)), 100. * correct / np.float(len(train_loader))

    def eval_on_task(self, test_loader, verbose=False):
        self.net.eval()
        correct = 0
        val_loss_classif = 0

        classe_prediction = np.zeros(10)
        classe_total = np.zeros(10)
        classe_wrong = np.zeros(10)  # Images wrongly attributed to a particular class

        for data, target in test_loader:
            batch = variable(data)
            label = variable(target.squeeze())
            classif = self.net(batch)
            loss_classif = F.nll_loss(classif, label)
            val_loss_classif += loss_classif.item()
            pred = classif.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

            for i in range(label.data.shape[0]):
                if pred[i].cpu()[0] == label.data[i].cpu():
                    classe_prediction[pred[i].cpu()[0]] += 1
                else:
                    classe_wrong[pred[i].cpu()[0]] += 1
                classe_total[label.data[i]] += 1

        val_loss_classif /= (np.float(len(test_loader.sampler)))
        valid_accuracy = 100. * correct / np.float(len(test_loader.sampler))

        if verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                val_loss_classif, correct, (len(test_loader)*self.batchsize),
                100. * correct / (len(test_loader)*self.batchsize)))

            for i in range(10):
                print('Classe {} Accuracy: {}/{} ({:.3f}%, Wrong : {})'.format(
                    i, classe_prediction[i], classe_total[i],
                    100. * classe_prediction[i] / classe_total[i], classe_wrong[i]))
            print('\n')

        return val_loss_classif, valid_accuracy, classe_prediction, classe_total, classe_wrong

    def forward(self, x, FID=False):
        return self.net.forward(x, FID)

    def save(self, ind_task, Best=False):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if Best:
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'Best_Classifier.pkl'))
        else:
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'Classifier_' + str(ind_task) + '.pkl'))

    def load_expert(self):

        expert_path = os.path.join(self.save_dir, '..', '..', '..', '..', '..', '..', '..', 'Classification',
                                   self.args.dataset,
                                   'Baseline', 'Num_tasks_1', 'seed_' + str(self.args.seed), 'Best_Classifier.pkl')

        if not os.path.exists(os.path.join(expert_path)):
            print('The expert does not exist, you can train it by running :')
            print(
                'python main.py --context Classification --task_type disjoint --method Baseline --dataset YOUR_DATASET --epochs 25 --num_task 1 --seed YOUR_SEED')

        self.net.load_state_dict(torch.load(expert_path))

    def labelize(self, batch, ind_task):
        self.net.eval()
        if self.gpu_mode:
            batch = batch.cuda(self.device)
        output = self.net(batch)
        return output[:, :ind_task + 1].max(dim=1)[1]

    def reinit(self):
        self.net.apply(Xavier)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.dropout = nn.Dropout(p=0.5)

        self.BN = nn.BatchNorm1d(320)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.apply(Xavier)

    def forward(self, x, FID=False):
        x = x.view(-1, 1, 28, 28)

        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.BN(x)
        if FID:
            return x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)
