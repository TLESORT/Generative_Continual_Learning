
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import os

def get_annotations_map(VAL_PATH):
    valAnnotationsPath = VAL_PATH + '/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]
    return valAnnotations


def get_image_folders(TRAIN_DIR):
    """
    Build an input pipeline for training and evaluation.
    For training data it does data augmentation.
    """

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    # intensities of enhancers
    factors = {
        0: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
        1: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        2: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        3: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
    }

    # randomly change color of an image
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        # random enhancers in random order
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image

    def rotate(image):
        degree = np.clip(np.random.normal(0.0, 15.0), -40.0, 40.0)
        return image.rotate(degree, Image.BICUBIC)

    # training data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.Lambda(rotate),
        #transforms.RandomCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # mean and std are taken from here:
    # http://pytorch.org/docs/master/torchvision/models.html
    train_folder = ImageFolder(TRAIN_DIR, train_transform)

    return train_folder

def get_test_image_folders(path):


    num_classes = 10
    TRAIN_DIR=path+'tiny-imagenet-200/training'
    VAL_DIR=path+'tiny-imagenet-200/validation'


    # for validation data
    val_transform = transforms.Compose([
        #transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_annotations_map = get_annotations_map(VAL_DIR)

    #val_folder = ImageFolder(VAL_DIR, val_transform)


    #X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype='uint8')
    #y_train = np.zeros([num_classes * 500], dtype='uint8')

    trainPath = TRAIN_DIR

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        '''
        for c in os.listdir(sChildPath):
            X = np.array(Image.open(os.path.join(sChildPath, c)))
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        '''
        j += 1
        if (j >= num_classes):
            break

    print('loading test images...')

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype='uint8')
    y_test = np.zeros([num_classes * 50], dtype='uint8')

    i = 0
    testPath = VAL_DIR + '/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = np.array(Image.open(sChildPath))
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    return DataTest(torch.from_numpy(X_test), torch.from_numpy(y_test)), y_test

class DataTest(Dataset):

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)

        print(data_tensor.type(torch.FloatTensor).shape)

        self.data_tensor = data_tensor.type(torch.FloatTensor)
        self.target_tensor = target_tensor.type(torch.LongTensor)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):

        return self.data_tensor.size(0)

# there is no annotation in this test set , therefor we can not use it for evaluation
'''
def get_test_image_folders(TEST_DIR):
    """
    Build an input pipeline for training and evaluation.
    For training data it does data augmentation.
    """


    # for validation data
    test_transform = transforms.Compose([
        #transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # mean and std are taken from here:
    # http://pytorch.org/docs/master/torchvision/models.html
    test_folder = ImageFolder(TEST_DIR, test_transform)
    return test_folder
'''