
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Classifiers.Classifier import Classifier, Net


class Fashion_Classifier(Classifier):

    def __init__(self, args):
        super(Fashion_Classifier, self).__init__(args)