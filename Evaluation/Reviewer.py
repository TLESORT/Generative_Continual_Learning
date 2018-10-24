import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import utils
from Classifiers.Fashion_Classifier import Fashion_Classifier
from Classifiers.Mnist_Classifier import Mnist_Classifier
from Classifiers.Cifar_Classifier import Cifar_Classifier
from Data.load_dataset import load_dataset_full, load_dataset_test, get_iter_dataset
from log_utils import *
from Data.data_loader import DataLoader
from Evaluation.tools import calculate_frechet_distance

mpl.use('Agg')


class Reviewer(object):
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
        self.trainEval = args.trainEval
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
        elif self.dataset == 'cifar10':
            self.Classifier = Cifar_Classifier(args)
        else:
            print('Not implemented')

    # this should be train on task
    def train_classifier(self, epoch, data_loader_train, ind_task):
        self.Classifier.net.train()

        train_loss_classif, train_accuracy = self.Classifier.train_on_task(data_loader_train, ind_task=ind_task,
                                                                           epoch=epoch,
                                                                           additional_loss=None)
        val_loss_classif, valid_accuracy, classe_prediction, classe_total, classe_wrong = self.Classifier.eval_on_task(
            self.valid_loader, self.verbose)

        if self.verbose:
            print(
                'Epoch: {} Train set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n Valid set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
                    epoch, train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy))
        return train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy, (
            100. * classe_prediction) / classe_total

    def compute_all_tasks_FID(self, args, Best=False):

        if Best:
            id = "Best_"
        else:
            id = ''

        list_FID = []

        for ind_task in range(self.num_task):
            list_FID.append(self.compute_FID(args, ind_task, Best))

        assert len(list_FID) == self.num_task

        list_FID = np.array(list_FID)

        np.savetxt(os.path.join(self.log_dir, id + 'Frechet_Inception_Distance_All_Tasks.txt'), list_FID)

    def compute_FID(self, args, ind_task, Best=False):

        if Best:
            id = "Best_"
        else:
            id = ''

        # load true data : upperbound_disjoint
        if 'upperbound' in self.task_type:
            test_file = self.task_type + '_' + str(self.num_task) + '_test.pt'
        else:
            test_file = 'upperbound_' + self.task_type + '_' + str(self.num_task) + '_test.pt'

        true_DataLoader = DataLoader(torch.load(os.path.join(self.data_dir, 'Tasks', self.dataset, test_file)), args)[
            self.num_task-1]


        # load generated data
        path = os.path.join(self.gen_dir, id + 'train_Task_' + str(ind_task) + '.pt')
        gen_DataLoader = DataLoader(torch.load(path), args)

        # compute FID
        return self.Frechet_Inception_Distance(gen_DataLoader, true_DataLoader, ind_task)

    def review(self, data_loader_train, task, Best=False):
        if Best:
            id = "Best_"
        else:
            id = ''

        if self.dataset == 'mnist':
            self.Classifier = Mnist_Classifier(self.args)
        elif self.dataset == 'fashion':
            self.Classifier = Fashion_Classifier(self.args)
        else:
            print('Not implemented')

        best_accuracy = -1
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        valid_acc = []
        valid_acc_classes = []

        path = os.path.join(self.sample_dir, id + 'samples4review_task_' + str(task) + '.png')

        if self.verbose:
            print("some sample from the generator")
        data_loader_train.visualize_sample(path, self.sample_num, [self.size, self.size, self.input_size])

        print("Task : " + str(task))

        early_stop = 0.
        # Training classifier
        for epoch in range(self.epoch_Review):
            tr_loss, tr_acc, v_loss, v_acc, v_acc_classes = self.train_classifier(epoch, data_loader_train, task)
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
            self.test_loader, self.verbose)

        test_acc_classes = 100. * classe_prediction / classe_total

        if self.verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy : ({:.2f}%)'.format(
                loss, test_acc))

            for i in range(10):
                print('Classe {} Accuracy: {}/{} ({:.3f}%, Wrong : {})'.format(
                    i, classe_prediction[i], classe_total[i],
                    100. * classe_prediction[i] / classe_total[i], classe_wrong[i]))

            print('\n')

        # loss, test_acc, test_acc_classes = self.test()  # self.test_classifier(epoch)
        np.savetxt(os.path.join(self.log_dir, id + 'data_classif_' + self.dataset + '-task' + str(task) + '.txt'),
                   np.transpose([train_loss, train_acc, val_loss, val_acc]))
        np.savetxt(os.path.join(self.log_dir, id + 'best_score_classif_' + self.dataset + '-task' + str(task) + '.txt'),
                   np.transpose([test_acc]))
        np.savetxt(
            os.path.join(self.log_dir, id + 'data_classif_classes' + self.dataset + '-task' + str(task) + '.txt'),
            np.transpose([test_acc_classes]))

        return valid_acc, valid_acc_classes

    def eval_on_train(self, data_loader_train, task):

        if self.dataset == 'mnist':
            self.Classifier = Mnist_Classifier(self.args)
        elif self.dataset == 'fashion':
            self.Classifier = Fashion_Classifier(self.args)
        else:
            print('Not implemented')

        self.Classifier.load_expert()
        self.Classifier.net.eval()
        print("trainEval Task : " + str(task))


        loss, train_acc, classe_prediction, classe_total, classe_wrong = self.Classifier.eval_on_task(data_loader_train, self.verbose)

        train_acc_classes = 100. * classe_prediction / classe_total

        if self.verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy : ({:.2f}%)'.format(
                loss, train_acc))

            for i in range(10):
                print('Classe {} Accuracy: {}/{} ({:.3f}%, Wrong : {})'.format(
                    i, classe_prediction[i], classe_total[i],
                    100. * classe_prediction[i] / classe_total[i], classe_wrong[i]))

            print('\n')

        return train_acc, train_acc_classes

    def eval_balanced_on_train(self, data_loader_train):

        cpt_classes = np.zeros(10)

        for i, (data, target) in enumerate(data_loader_train):

            for i in range(target.shape[0]):

                cpt_classes[target[i]] += 1

        print(cpt_classes.astype(int))
        return cpt_classes.astype(int)


    def review_all_tasks(self, args, Best=False):

        # before launching the programme we check that all files are here to nnot lose time
        for i in range(self.num_task):
            if Best:  # Best can be use only for Baseline
                path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else:
                path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')

            assert os.path.isfile(path)

        for i in range(self.num_task):

            if Best:  # Best can be use only for Baseline
                path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else:
                path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')

            data_loader_train = DataLoader(torch.load(path), args)

            self.review(data_loader_train, i, Best)

    def review_all_trainEval(self, args, Best=False):

        if Best:
            id = "Best_"
        else:
            id = ''

        list_trainEval = []
        list_trainEval_classes = []
        list_balance_classes = []

        # before launching the programme we check that all files are here to nnot lose time
        for i in range(self.num_task):
            if Best:  # Best can be use only for Baseline
                path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else:
                path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')
            assert os.path.isfile(path)

        for i in range(self.num_task):
            if Best:  # Best can be use only for Baseline
                path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else:
                path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')

            data_loader_train = DataLoader(torch.load(path), args)

            if self.conditional or Best:
                train_acc, train_acc_classes = self.eval_on_train(data_loader_train, self.verbose)
                list_trainEval.append(train_acc)
                list_trainEval_classes.append(train_acc_classes)
            else:
                classe_balance = self.eval_balanced_on_train(data_loader_train)
                list_balance_classes.append(classe_balance)

        if self.conditional or Best:
            assert len(list_trainEval) == self.num_task

            list_trainEval = np.array(list_trainEval)
            list_trainEval_classes = np.array(list_trainEval)

            np.savetxt(os.path.join(self.log_dir, id + 'TrainEval_All_Tasks.txt'), list_trainEval)
            np.savetxt(os.path.join(self.log_dir, id + 'TrainEval_classes_All_Tasks.txt'), list_trainEval_classes)
        else:
            assert len(list_balance_classes) == self.num_task
            np.savetxt(os.path.join(self.log_dir, id + 'Balance_classes_All_Tasks.txt'), list_balance_classes)

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

    def Frechet_Inception_Distance(self, Gen_DataLoader, True_DataLoader, ind_task):

        eval_size = 50

        # 0. load reference classifier

        # self.load_best_baseline()  # we load the best classifier

        self.Classifier.load_expert()

        self.Classifier.net.eval()
        if self.dataset == "mnist":
            latent_size = 320
        elif self.dataset == "fashion":
            latent_size = 320

        real_output_table = torch.FloatTensor(eval_size * self.batch_size, latent_size)
        gen_output_table = torch.FloatTensor(eval_size * self.batch_size, latent_size)

        # print("get activations on test data")
        for i, (data, target) in enumerate(True_DataLoader):
            if i >= eval_size or i >= (
                        int(len(True_DataLoader) / self.batch_size) - 1):  # (we throw away the last batch)
                break
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            label = Variable(target.squeeze())
            activation = self.Classifier.net(batch, FID=True)

            real_output_table[i * self.batch_size:(i + 1) * self.batch_size, :] = activation.data

        # print("get activations on generated data")
        Gen_DataLoader.shuffle_task()
        for i, (data, target) in enumerate(Gen_DataLoader):

            if i >= eval_size or i >= (
                        int(len(Gen_DataLoader) / self.batch_size) - 1):  # (we throw away the last batch)
                break

            # 2. use the reference classifier to compute the output vector
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            label = Variable(target.squeeze())
            activation = self.Classifier.net(batch, FID=True)
            gen_output_table[i * self.batch_size:(i + 1) * self.batch_size, :] = activation.data

        # compute mu_real and sigma_real

        mu_real = real_output_table.cpu().numpy().mean(0)
        cov_real = np.cov(real_output_table.cpu().numpy().transpose())

        assert mu_real.shape[0] == latent_size
        assert cov_real.shape[0] == cov_real.shape[1] == latent_size

        mu_gen = gen_output_table.cpu().numpy().mean(0)
        cov_gen = np.cov(gen_output_table.cpu().numpy().transpose())

        assert mu_gen.shape[0] == latent_size
        assert cov_gen.shape[0] == cov_gen.shape[1] == latent_size

        Frechet_Inception_Distance = calculate_frechet_distance(mu_real, cov_real, mu_gen, cov_gen)

        if self.verbose:
            print("Frechet Inception Distance")
            print(Frechet_Inception_Distance)

        return Frechet_Inception_Distance
