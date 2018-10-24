

from Classifiers.Classifier import Classifier, Net

class Mnist_Classifier(Classifier):
    def __init__(self, args):
        super(Mnist_Classifier, self).__init__(args)

class Mnist_Net(Net):

    def __init__(self):
        super(Mnist_Net, self).__init__()