from Training.Trainer import Trainer
from Data.data_loader import DataLoader


class Rehearsal(Trainer):
    def __init__(self, model, args):
        super(Rehearsal, self).__init__(model, args)
        self.nb_samples_reharsal = args.nb_samples_reharsal
        self.data_memory = None
        self.task_samples = None
        self.task_labels = None

    def create_next_data(self, ind_task):
        #
        # save sample before modification of training set
        x_tr, y_tr = self.train_loader[ind_task].get_sample(self.nb_samples_reharsal)
        if self.gpu_mode:
            x_tr, y_tr = x_tr.cpu(), y_tr.cpu()
        self.task_samples = x_tr.clone()
        self.task_labels = y_tr.clone()

        # create data loder with memory fro; previous task
        if ind_task > 0:

            # balanced the number of sample and incorporate it in the memory

            # put the memory inside the training dataset
            self.train_loader[ind_task].concatenate(self.data_memory)
            self.train_loader[ind_task].shuffle_task()
            train_loader = self.train_loader[ind_task]
            test_loader = None

        else:
            train_loader = self.train_loader[ind_task]
            test_loader = None
            # test_loader = self.test_loader[ind_task]

        # Add data to memory at the end
        c1 = 0
        c2 = 1
        tasks_tr = []  # reset the list

        # save samples from the actual task in the memory

        tasks_tr.append([(c1, c2), self.task_samples.clone().view(-1, 784), self.task_labels.clone().view(-1)])
        increase_factor = int(self.sample_transfer / self.nb_samples_reharsal)
        if ind_task <= 0:
            self.data_memory = DataLoader(tasks_tr, self.args).increase_size(increase_factor)
        else:
            self.data_memory.concatenate(DataLoader(tasks_tr, self.args).increase_size(increase_factor))

        return train_loader, test_loader
