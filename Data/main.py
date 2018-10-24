import argparse
import os

# from CUB200 import CUB200_Disjoint
# from cifar100 import Cifar100_Disjoint
from disjoint import Disjoint
from rotations import Rotations
from permutations import Permutations
from fashion import fashion
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

import torch

parser = argparse.ArgumentParser()

parser.add_argument('--dir', default='../Archives', help='input directory')
parser.add_argument('--i', default='Data', help='input directory')
parser.add_argument('--train_file', default='', help='input directory')
parser.add_argument('--test_file', default='', help='input directory')

parser.add_argument('--upperbound', default=False, type=bool)
parser.add_argument('--task', default='disjoint', choices=['rotations', 'permutations',
                                                           'disjoint', 'cifar100', 'CUB200'],
                    help='type of task to create', )
parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist', 'fashion', 'cifar10'])
parser.add_argument('--n_tasks', default=3, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='input batch size')
parser.add_argument('--min_rot', default=0., type=float, help='minimum rotation')
parser.add_argument('--max_rot', default=90., type=float, help='maximum rotation')
args = parser.parse_args()

torch.manual_seed(args.seed)


print(str(args).replace(',', ',\n'))


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


args.i = os.path.join(args.dir, args.i)
args.o = os.path.join(args.i, 'Tasks', args.dataset)
args.i = os.path.join(args.i, 'Datasets', args.dataset)
args.train_file = 'training.pt'
args.test_file = 'test.pt'

# download data if possible
if args.dataset == 'mnist':
    datasets.MNIST(args.i, train=True, download=True, transform=transforms.ToTensor())
elif args.dataset == 'fashion':
    fashion(args.i, train=True, download=True, transform=transforms.ToTensor())
elif args.dataset == 'cifar10':
    print("DL one later")
elif args.dataset == 'cifar100':
    args.train_file = 'cifar100.pt'
    if not os.path.isdir(args.i):
        print('This dataset should be downloaded manually')
elif args.dataset == 'CUB200':
    args.i = args.i = os.path.join(args.i, 'images')
    if not os.path.isdir(args.i):
        print('This dataset should be downloaded manually')

if not os.path.exists(args.o):
    os.makedirs(args.o)

args.i = os.path.join(args.i, 'processed')

if args.task == 'rotations':
    DataFormatter = Rotations(args)
elif args.task == 'permutations':
    DataFormatter = Permutations(args)
elif args.task == 'disjoint':
    DataFormatter = Disjoint(args)
elif args.task == 'cifar100':
    DataFormatter = Cifar100_Disjoint(args)
elif args.task == 'CUB200':
    DataFormatter = CUB200_Disjoint(args)
else:
    print("Not Implemented")

DataFormatter.formating_data()
