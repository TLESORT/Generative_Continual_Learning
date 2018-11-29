from Generative_Models.Generative_Model import GenerativeModel
import torch
from utils import *


class ConditionalModel(GenerativeModel):


    # if no task2generate are given we generate all labellize for all task
    # if task2generate and annotate == false we generate only for the actual task
    # if task2generate and annotate == true we generate only for all past tasks
    def sample(self, batch_size, task2generate=None, multi_annotation=False):
        '''
        :param batch_size:
        :param task2generate: give the index of class to generate (the name is a bit misleading)
        :param multi_annotation: indicate if we want just one classes or all classes <= task2generate
        :param expert: classifier that can give a label to samples
        :return: batch of  sample from different classes and return a batch of images and label
        '''

        self.G.eval()

        if task2generate is not None:
            classes2generate=task2generate + 1
        else:
            classes2generate=self.num_classes

        z_ = self.random_tensor(batch_size, self.z_dim)
        if multi_annotation:
            # keep this please
            # y = torch.LongTensor(batch_size, 1).random_() % self.num_classes
            y = (torch.randperm(batch_size * 10) % classes2generate)[:batch_size]
            y_onehot = self.get_one_hot(y)
        else:
            y = (torch.ones(batch_size) * (classes2generate-1)).long()
            y_onehot = self.get_one_hot(y).cuda()

        output = self.G(variable(z_), y_onehot).data

        return output, y

    # For conditional Replay we generate tasks one by one
    def generate_batch4Task(self, nb_sample_train, task2generate, multi_annotation):
        return self.sample(batch_size=nb_sample_train, task2generate=task2generate, multi_annotation=False)



    def get_one_hot(self, y):
        y_onehot = torch.FloatTensor(y.shape[0], self.num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, y[:, np.newaxis], 1.0)

        return y_onehot


    # This function generate a dataset for one class or for all class until ind_task included
    def generate_dataset(self, ind_task, nb_sample_per_task, one_task=True, Train=True, classe2generate=None):

        # to generate 10 classes classe2generate is 9 as classes 0 to 9
        if classe2generate is not None:
            assert classe2generate <= self.num_classes
            if self.task_type != "disjoint":
                assert classe2generate == self.num_classes
        else:
            classe2generate = ind_task+1

        train_loader_gen=None

        if Train:
            path = os.path.join(self.gen_dir, 'train_Task_' + str(ind_task) + '.pt')
            path_samples = os.path.join(self.sample_dir, 'samples_train_' + str(ind_task) + '.png')
        else:
            path = os.path.join(self.gen_dir, 'test_Task_' + str(ind_task) + '.pt')
            path_samples = os.path.join(self.sample_dir, 'samples_test_' + str(ind_task) + '.png')

        # if we have only on task to generate
        if one_task or classe2generate == 0:  # generate only for the task ind_task

            train_loader_gen = self.generate_task(nb_sample_per_task, multi_annotation=False, classe2generate=classe2generate)

        else:  # else case we generate for all previous task

            for i in range(classe2generate):  # we take from all task, actual one included

                train_loader_ind = self.generate_task(nb_sample_per_task, multi_annotation=True, classe2generate=i)

                if i == 0:
                    train_loader_gen = train_loader_ind
                else:
                    train_loader_gen.concatenate(train_loader_ind)

        # we save the concatenation of all generated with the actual task for train and test
        train_loader_gen.save(path)
        train_loader_gen.visualize_sample(path_samples, self.sample_num, [self.size, self.size, self.input_size])

        # return the the train loader with all data
        return train_loader_gen  # test_loader_gen # for instance we don't use the test set

    # this generation only works for Baseline, disjoint
    # we generate the dataset based on one generator by task to get normally the best generated dataset
    # can be used to generate train or test data
    def generate_best_dataset(self, ind_task, nb_sample_per_task, one_task=True, Train=True, classe2generate=None):


        # to generate 10 classes classe2generate is 9 as classes 0 to 9
        if classe2generate is not None:
            assert classe2generate <= self.num_classes
            if self.task_type != "disjoint":
                assert classe2generate == self.num_classes
        else:
            classe2generate = ind_task+1

        if Train:
            path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(ind_task) + '.pt')
        else:
            path = os.path.join(self.gen_dir, 'Best_test_Task_' + str(ind_task) + '.pt')

        # if we have only on task to generate
        if one_task or classe2generate == 0:  # generate only for the task ind_task
            # we do not need automatic annotation since we have one generator by class
            previous_data_train = self.generate_task(nb_sample_per_task, multi_annotation=False, classe2generate=classe2generate)
            #previous_data_train = DataLoader(tasks_tr, self.args)

        else:  # else we load the previous dataset and add the new data

            previous_path_train = os.path.join(self.gen_dir, 'Best_train_Task_' + str(ind_task - 1) + '.pt')

            previous_data_train = DataLoader(torch.load(previous_path_train), self.args)

            # we do not need automatic annotation since we have one generator by class
            train_loader_ind = self.generate_task(nb_sample_per_task, multi_annotation=False, classe2generate=i)

            previous_data_train.concatenate(train_loader_ind)

        # we save the concatenation of all generated with the actual task for train and test
        previous_data_train.save(path)

        # return nothing