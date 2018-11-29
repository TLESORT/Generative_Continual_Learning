

from utils import *
from log_utils import *
from tqdm import tqdm
from Data.data_loader import DataLoader
import time
import numpy as np
from Evaluation.Reviewer import Reviewer


class Trainer(object):
    def __init__(self,model, args, reviewer=None):
        self.args=args

        self.context=args.context

        if self.context=="Generation":
            self.reviewer = reviewer

        self.conditional = args.conditional
        self.dataset = args.dataset
        self.batch_size=args.batch_size
        self.gpu_mode=args.gpu_mode
        self.device=args.device
        self.verbose=args.verbose

        if self.dataset=="mnist" or self.dataset=="fashion":
            self.image_size = 28
            self.input_size = 1
        elif self.dataset=="cifar10":
            self.image_size = 32
            self.input_size = 3

        self.model=model

        self.sample_dir = args.sample_dir
        self.sample_transfer = args.sample_transfer

        self.sample_num = 100
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.task_type = args.task_type
        self.method = args.method
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.epochs_gan = args.epoch_G
        self.task = None
        self.old_task = None
        self.num_task = args.num_task
        self.num_classes = args.num_classes
        self.ind_task=0
        self.samples_per_task=args.samples_per_task

        train_loader, test_loader, n_inputs, n_outputs, n_tasks = load_datasets(args)
        self.train_loader = DataLoader(train_loader, args)
        self.test_loader = DataLoader(test_loader, args)


    def forward(self, x, ind_task):
        return self.model.net.forward(x)

    def additional_loss(self, model):
        return None

    def create_next_data(self, ind_task):
        return self.train_loader[ind_task], self.test_loader[ind_task]

    def preparation_4_task(self, model, ind_task):

        if ind_task > 0 and self.context == 'Generation':
            # We generate as much image as there is in the actual task

            #nb_sample_train = len(self.train_loader[ind_task])
            print("numbe of train sample is fixed as : " + str(self.sample_transfer))
            nb_sample_train = self.sample_transfer # approximate size of one task
            nb_sample_test = int(nb_sample_train * 0.2) # not used in fact

            # we generate dataset for later evaluation with image from previous tasks
            self.model.generate_dataset(ind_task - 1, nb_sample_train, one_task=False, Train=True)
            self.model.generate_dataset(ind_task - 1, nb_sample_test, one_task=False, Train=False)

        train_loader, test_loader = self.create_next_data(ind_task)
        return train_loader, test_loader

    def run_generation_tasks(self):
        # run each task for a model

        self.model.G.apply(self.model.G.weights_init)
        loss, acc, acc_all_tasks = {}, {}, {}
        timestamp = time.time()
        log_time = []
        for ind_task in range(self.args.num_task):

            print("Task : " + str(ind_task))

            if 'Ewc' in self.method:
                train_loader, test_loader = self.preparation_4_task(self.model, ind_task)
            else:
                train_loader, test_loader = self.preparation_4_task(self.model.G, ind_task)
            self.ind_task=ind_task

            #self.visualize_Samples(train_loader, ind_task)

            path = os.path.join(self.sample_dir, 'sample_' + str(ind_task) + '.png')

            if self.verbose:
                print("some sample from the train_laoder")
            self.train_loader.visualize_sample(path, self.sample_num, [self.image_size, self.image_size, self.input_size])

            loss[ind_task] = []
            acc[ind_task] = []
            acc_all_tasks[ind_task] = []
            start_time = time.time()


            for epoch in range(self.args.epochs):
                print("Epoch : "+ str(epoch))

                loss_epoch = self.model.train_on_task(train_loader, ind_task, epoch, self.additional_loss)

                self.model.visualize_results((epoch + 1), ind_task)

                loss[ind_task].append(loss_epoch)

                # Eval the FID

                '''
                if 'upperbound' in self.task_type:
                    test_file = self.task_type + '_' + str(self.num_task) + '_test.pt'
                else:
                    test_file = 'upperbound_' + self.task_type + '_' + str(self.num_task) + '_test.pt'
                gen_DataLoader = self.model.generate_task(ind_task, nb_sample_train=1000)
                true_DataLoader = self.test_loader[ind_task]
                FID_epoch = self.reviewer.Frechet_Inception_Distance(gen_DataLoader, true_DataLoader, ind_task)
                '''

                #for previous_task in range(ind_task + 1):
                    #acc[previous_task].append(self.test_G(previous_task))
                #acc_all_tasks[ind_task].append(self.test_G_all_tasks())
            # Or save generator
            self.model.save_G(self.ind_task)

            log_time.append(time.time() - timestamp)
            timestamp = time.time()

        np.savetxt(os.path.join(self.log_dir,'task_training_time.txt'), log_time)


        #nb_sample_train = len(self.train_loader[0])
        nb_sample_train = self.sample_transfer  # approximate size of one task
        nb_sample_test = int(nb_sample_train * 0.2)

        # generate dataset for all task (indice of last task is num_task-1) and save it
        self.model.generate_dataset(self.num_task-1, nb_sample_train, one_task=False, Train=True)
        #self.model.generate_dataset(self.num_task-1, nb_sample_test, one_task=False, Train=False)

        if self.method == 'Baseline': # this kind of thing should not exist, inheritance should avoid it
            self.model.generate_best_dataset(self.num_task - 1, nb_sample_train, Train=True)
            #self.model.generate_best_dataset(self.num_task - 1, nb_sample_test, Train=True)



    def run_classification_tasks(self):
        accuracy_test = 0
        loss, acc, acc_all_tasks = {}, {}, {}
        for ind_task in range(self.num_task):
            accuracy_task = 0
            train_loader, test_loader = self.preparation_4_task(self.model.net, ind_task)

            self.ind_task=ind_task

            if not self.args.task_type == "CUB200":
                path = os.path.join(self.sample_dir, 'sample_' + str(ind_task) + '.png')

                if self.verbose:
                    print("some sample from the train_loader")
                self.train_loader.visualize_sample(path, self.sample_num, [self.image_size, self.image_size, self.input_size])
            else:
                print("visualisation of CUB200 not implemented")
            loss[ind_task] = []
            acc[ind_task] = []
            acc_all_tasks[ind_task] = []
            for epoch in tqdm(range(self.args.epochs)):
                loss_epoch, accuracy_epoch = self.model.train_on_task(train_loader, ind_task, epoch, self.additional_loss)
                loss[ind_task].append(loss_epoch)

                if accuracy_epoch > accuracy_task:
                    self.model.save(ind_task)
                    accuracy_task = accuracy_epoch

                for previous_task in range(ind_task + 1):
                    loss_test, test_acc, classe_prediction, classe_total, classe_wrong = self.model.eval_on_task(
                        self.test_loader[previous_task], 0)

                    #acc[previous_task].append(self.test(previous_task))
                    acc[previous_task].append(test_acc)

                accuracy_test_epoch=self.test_all_tasks()
                acc_all_tasks[ind_task].append(accuracy_test_epoch)

                if accuracy_test_epoch > accuracy_test:
                #if True:
                    self.model.save(ind_task, Best=True)
                    accuracy_test = accuracy_test_epoch


        loss_plot(loss, self.args)
        accuracy_plot(acc, self.args)
        accuracy_all_plot(acc_all_tasks, self.args)


    def test_all_tasks(self):
        self.model.net.eval()

        mean_task = 0
        if self.task_type == 'upperbound':
            loss, mean_task, classe_prediction, classe_total, classe_wrong = self.model.eval_on_task(
                self.test_loader[self.num_task - 1], 0)
        else:
            for ind_task in range(self.num_task):
                loss, acc_task, classe_prediction, classe_total, classe_wrong = self.model.eval_on_task(
                    self.test_loader[ind_task], 0)

                mean_task += acc_task
            mean_task = mean_task/self.num_task
        print("Mean overall performance : " + str(mean_task.item()))
        return mean_task


    def regenerate_datasets_for_eval(self):

        nb_sample_train = self.sample_transfer  #len(self.train_loader[0])
        #nb_sample_test = int(nb_sample_train * 0.2)

        for i in range(self.args.num_task):
            self.model.load_G(ind_task=i)
            self.generate_dataset(i, nb_sample_train, classe2generate=i+1, Train=True)

        return

    def generate_dataset(self, ind_task,sample_per_classes, classe2generate, Train=True):
        return self.model.generate_dataset(ind_task, sample_per_classes, one_task=False, Train=Train, classe2generate=classe2generate)

