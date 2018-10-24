import argparse
import os.path
import torch

import numpy as np

from torchvision import datasets, transforms


class Disjoint(object):
    def __init__(self, args):
        super(Disjoint, self).__init__()

        self.upperbound = args.upperbound
        self.n_tasks = args.n_tasks
        self.i = args.i
        self.train_file = args.train_file
        self.test_file = args.test_file
        self.dataset = args.dataset

        if self.upperbound:
            self.o_train = os.path.join(args.o, 'upperbound_disjoint_' + str(self.n_tasks) + '_train.pt')
            self.o_test = os.path.join(args.o, 'upperbound_disjoint_' + str(self.n_tasks) + '_test.pt')
        else:
            self.o_train = os.path.join(args.o, 'disjoint_' + str(self.n_tasks) + '_train.pt')
            self.o_test = os.path.join(args.o, 'disjoint_' + str(self.n_tasks) + '_test.pt')


    def load_cifar10(self):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10(root='./Datasets', train=True, download=True, transform=transform_train)
        tensor_data = torch.Tensor(len(dataset_train),3,32,32)
        tensor_label = torch.LongTensor(len(dataset_train))

        for i in range(len(dataset_train)):
            tensor_data[i] = dataset_train[i][0]
            tensor_label[i] = dataset_train[i][1]

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        tensor_test = torch.Tensor(len(dataset_test),3,32,32)
        tensor_label_test = torch.LongTensor(len(dataset_test))

        for i in range(len(dataset_test)):
            tensor_test[i] = dataset_test[i][0]
            tensor_label_test[i] = dataset_test[i][1]

        #testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        return tensor_data, tensor_label, tensor_test, tensor_label_test



    def formating_data(self):



        tasks_tr = []
        tasks_te = []

        if  self.dataset == 'cifar10':
            x_tr, y_tr, x_te, y_te = self.load_cifar10()

            x_tr = x_tr.float().view(x_tr.size(0), -1)
            x_te = x_te.float().view(x_te.size(0), -1)
        else:
            assert os.path.isfile(os.path.join(self.i, self.train_file))
            assert os.path.isfile(os.path.join(self.i, self.test_file))

            x_tr, y_tr = torch.load(os.path.join(self.i, self.train_file))
            x_te, y_te = torch.load(os.path.join(self.i, self.test_file))

            x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
            x_te = x_te.float().view(x_te.size(0), -1) / 255.0

        y_tr = y_tr.view(-1).long()
        y_te = y_te.view(-1).long()

        cpt = int(10 / self.n_tasks)

        for t in range(self.n_tasks):
            if self.upperbound:
                c1 = 0
            else:
                c1 = t * cpt
            c2 = (t + 1) * cpt
            i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
            i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
            tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
            tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

        torch.save(tasks_tr, self.o_train)
        torch.save(tasks_te, self.o_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i', default='raw/cifar100.pt', help='input directory')
    parser.add_argument('--o', default='cifar100.pt', help='output file')
    parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    DataFormater = Disjoint()
    DataFormater.formating_data(args)
