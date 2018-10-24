import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

import utils
from Data.fashion import fashion
from Data.input_pipeline import get_image_folders, get_test_image_folders


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def load_dataset_full(data_dir, dataset, num_examples=50000):

    fas=False
    path = os.path.join(data_dir, 'Datasets', dataset)

    if dataset == 'mnist':
        dataset = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
        dataset_train = Subset(dataset, range(num_examples))
        dataset_val = Subset(dataset, range(50000, 60000))
    elif dataset == 'fashion':
        if fas:
            dataset = datasets.FashionMNIST(path, train=True, download=True, transform=transforms.ToTensor())
        else:

            dataset = fashion(path, train=True, download=True, transform=transforms.ToTensor())
        dataset_train = Subset(dataset, range(num_examples))
        dataset_val = Subset(dataset, range(50000, 60000))
    elif dataset == 'cifar10':
        if num_examples > 45000: num_examples = 45000 # does not work if num_example > 50000
        transform = transforms.Compose(
                [transforms.ToTensor()])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        dataset_train = Subset(dataset, range(num_examples))
        dataset_val = Subset(dataset, range(45000, 50000))
    elif dataset == 'lsun':
        transform = transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        dataset_train = datasets.LSUN(db_path=path+'/LSUN/', classes=['bedroom_train', 'bridge_train', 'church_outdoor_train', 'classroom_train',
            'conference_room_train', 'dining_room_train', 'kitchen_train',
            'living_room_train', 'restaurant_train', 'tower_train'],transform=transform)

        dataset_val = datasets.LSUN(db_path=path+'/LSUN/', classes=['bedroom_val', 'bridge_val', 'church_outdoor_val', 'classroom_val',
            'conference_room_val', 'dining_room_val', 'kitchen_val',
            'living_room_val', 'restaurant_val', 'tower_val'],transform=transform)
    elif dataset == 'timagenet':
        dataset = get_image_folders(path+'tiny-imagenet-200/training')

        size = len(dataset)
        indices = torch.randperm(size)

        dataset_train = Subset(dataset, indices[:int(size*0.8)])
        dataset_val = Subset(dataset, indices[int(size*0.8):])



    list_classes_train = np.asarray([dataset_train[i][1] for i in range(len(dataset_train))])
    list_classes_val = np.asarray([dataset_val[i][1] for i in range(len(dataset_val))])

    if dataset == 'timagenet':
        #we only use 10 classes in the dataset
        list_classes_train = np.where(list_classes_train < 10)[0]
        list_classes_val = np.where(list_classes_val < 10)[0]

        dataset_train = Subset(dataset_val, list_classes_train)
        dataset_val = Subset(dataset_val, list_classes_train)

    return dataset_train, dataset_val, list_classes_train, list_classes_val



def load_dataset_test(data_dir, dataset, batch_size):
    list_classes_test = []

    fas=False

    path = os.path.join(data_dir, 'Datasets', dataset)
    
    if dataset == 'mnist':
        dataset_test = datasets.MNIST(path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset == 'fashion':
        if fas:
            dataset_test = DataLoader(
                datasets.FashionMNIST(path, train=False, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()])),
                batch_size=batch_size)
        else:
            dataset_test = fashion(path, train=False, download=True, transform=transforms.ToTensor())

    elif dataset == 'cifar10':
        transform = transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset_test = datasets.CIFAR10(root=path, train=False,
                   download=True, transform=transform)

    elif dataset == 'celebA':
        dataset_test = utils.load_celebA(path + 'celebA', transform=transforms.Compose(
            [transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()]), batch_size=batch_size)
    elif dataset == 'timagenet':
        dataset_test, labels = get_test_image_folders(path)
        list_classes_test = np.asarray([labels[i] for i in range(len(dataset_test))])
        dataset_test = Subset(dataset_test, np.where(list_classes_test < 10)[0])
        list_classes_test = np.where(list_classes_test < 10)[0]

    list_classes_test = np.asarray([dataset_test[i][1] for i in range(len(dataset_test))])

    return dataset_test, list_classes_test


def get_iter_dataset(dataset, list_classe=[], batch_size=64, classe=None):
    if classe is not None:
        dataset = Subset(dataset, np.where(list_classe == classe)[0])

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader
