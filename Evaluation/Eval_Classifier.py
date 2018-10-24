import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import utils
from Classifiers.Fashion_Classifier import Fashion_Classifier
from Classifiers.Mnist_Classifier import Mnist_Classifier
from Data.load_dataset import load_dataset_full, load_dataset_test, get_iter_dataset
from log_utils import *
from Data.data_loader import DataLoader
from Evaluation.tools import calculate_frechet_distance

mpl.use('Agg')


class Reviewer_C(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.epoch_Review = args.epoch_Review
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.sample_dir = args.sample_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.data_dir = args.data_dir
        self.gen_dir = args.gen_dir
        self.verbose = args.verbose

        self.lr = args.lrC
        self.momentum = args.momentum
        self.log_interval = 100
        self.sample_num = 100
        self.size_epoch = args.size_epoch
        self.gan_type = args.gan_type
        self.conditional = args.conditional
        self.device = args.device
        self.TrainEval = args.TrainEval
        self.num_task = args.num_task
        self.task_type = args.task_type
        self.context = args.context

        self.seed = args.seed

        if self.conditional:
            self.model_name = 'C' + self.model_name

        # Load the generator parameters

        # The reviewer evaluate generate dataset (loader train) on true data (loader test)
        # not sur yet if valid should be real or not (it was before)
        dataset_train, dataset_valid, list_class_train, list_class_valid = load_dataset_full(self.data_dir,
                                                                                             args.dataset)
        dataset_test, list_class_test = load_dataset_test(self.data_dir, args.dataset, args.batch_size)

        # create data loader for validation and testing
        self.valid_loader = get_iter_dataset(dataset_valid)
        self.test_loader = get_iter_dataset(dataset_test)

        if self.dataset == 'mnist':
            self.input_size = 1
            self.size = 28
        elif self.dataset == 'fashion':
            self.input_size = 1
            self.size = 28
        elif self.dataset == 'cifar10':
            self.input_size = 3
            self.size = 32

        if self.dataset == 'mnist':
            self.Classifier = Mnist_Classifier(args)
        elif self.dataset == 'fashion':
            self.Classifier = Fashion_Classifier(args)
        else:
            print('Not implemented')

    # this should be train on task
    def train_classifier(self, epoch, data_loader_train, ind_task):
        self.Classifier.net.train()

        train_loss_classif, train_accuracy = self.Classifier.train_on_task(data_loader_train, ind_task=ind_task, epoch=epoch,
                                                                           additional_loss=None)
        val_loss_classif, valid_accuracy, classe_prediction, classe_total, classe_wrong = self.Classifier.eval_on_task(
            self.valid_loader, epoch)

        if self.verbose:
            print(
                'Epoch: {} Train set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n Valid set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
                    epoch, train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy))
        return train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy, (
        100. * classe_prediction) / classe_total



    def review(self, data_loader_train, value):


        self.Classifier.reinit()

        best_accuracy = -1
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        valid_acc = []
        valid_acc_classes = []

        print("Number of samples : " + str(value))

        early_stop = 0.
        # Training classifier
        for epoch in range(self.epoch_Review):
            tr_loss, tr_acc, v_loss, v_acc, v_acc_classes = self.train_classifier(epoch, data_loader_train, value)
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            # Save best model
            if v_acc > best_accuracy:
                if self.verbose:
                    print("New Best Classifier")
                    print(v_acc)
                best_accuracy = v_acc
                self.save(best=True)
                early_stop = 0.
            if early_stop == 60:
                break
            else:
                early_stop += 1
            valid_acc.append(np.array(v_acc))
            valid_acc_classes.append(np.array(v_acc_classes))

        # Then load best model
        self.load()

        loss, test_acc, classe_prediction, classe_total, classe_wrong = self.Classifier.eval_on_task(
            self.test_loader, 0)



        test_acc_classes = 100. * classe_prediction / classe_total

        if self.verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy : ({:.2f}%)'.format(
                loss, test_acc ))

            for i in range(10):
                print('Classe {} Accuracy: {}/{} ({:.3f}%, Wrong : {})'.format(
                    i, classe_prediction[i], classe_total[i],
                    100. * classe_prediction[i] / classe_total[i], classe_wrong[i]))

            print('\n')


        # loss, test_acc, test_acc_classes = self.test()  # self.test_classifier(epoch)
        np.savetxt(os.path.join(self.log_dir,'data_classif_' + self.dataset + '-num_samples_' + str(value) + '.txt'),
                   np.transpose([train_loss, train_acc, val_loss, val_acc]))
        np.savetxt(os.path.join(self.log_dir,'best_score_classif_' + self.dataset + '-num_samples_' + str(value) + '.txt'),
                   np.transpose([test_acc]))
        np.savetxt(os.path.join(self.log_dir,'data_classif_classes' + self.dataset + '-num_samples_' + str(value) + '.txt'),
                   np.transpose([test_acc_classes]))


        return valid_acc, valid_acc_classes

    def review_all_tasks(self, args, list_values):
        for value in list_values:
            # create data set with value samples
            dataset_train, dataset_valid, list_class_train, list_class_valid = load_dataset_full(self.data_dir,
                                                                                                 args.dataset, value)

            data_loader_train = get_iter_dataset(dataset_train)
            #data_loader_train = DataLoader(dataset_train, args)

            self.review(data_loader_train, value)

    # save a classifier or the best classifier
    def save(self, best=False):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if best:
            torch.save(self.Classifier.net.state_dict(),
                       os.path.join(self.save_dir, self.model_name + '_Classifier_Best.pkl'))
        else:
            torch.save(self.Classifier.net.state_dict(),
                       os.path.join(self.save_dir, self.model_name + '_Classifier.pkl'))

    # load the best classifier or the reference classifier trained on true data only
    def load(self, reference=False):
        if reference:
            save_dir = os.path.join(self.save_dir, "..", "..", "..", "Classifier", 'seed_' + str(self.seed))
            self.Classifier.net.load_state_dict(torch.load(os.path.join(save_dir, 'Classifier_Classifier_Best.pkl')))
        else:
            self.Classifier.net.load_state_dict(
                torch.load(os.path.join(self.save_dir, self.model_name + '_Classifier_Best.pkl')))

    def load_best_baseline(self):

        # best seed searched in the list define in get_best_baseline function, liste_seed = [1, 2, 3, 4, 5, 6, 7, 8]
        best_seed = utils.get_best_baseline(self.log_dir, self.dataset)

        save_dir = os.path.join(self.save_dir, "..", "..", "..", "Classifier", 'seed_' + str(best_seed))
        self.Classifier.net.load_state_dict(torch.load(os.path.join(save_dir, 'Classifier_Classifier_Best.pkl')))
