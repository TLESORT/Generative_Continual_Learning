from Training.Trainer import Trainer


class Generative_Replay(Trainer):
    def __init__(self, model, args):
        super(Generative_Replay, self).__init__(model, args)

    def create_next_data(self, ind_task):

        task_te_gen = None

        if ind_task > 0:

            #nb_sample_train = len(self.train_loader[ind_task])
            #nb_sample_test = int(nb_sample_train * 0.2)

            self.train_loader[ind_task] #we set the good index of dataset
            #self.test_loader[ind_task] #we set the good index of dataset

            #print("nb_sample_train should be higher than this")
            print("numbe of train sample per task is fixed as : " + str(self.sample_transfer))

            nb_sample_train = self.sample_transfer # approximate size of one task
            nb_sample_test = int(nb_sample_train * 0.2)

            task_tr_gen = self.model.generate_dataset(ind_task - 1, nb_sample_train, one_task=False, Train=True)
            #task_tr_gen = self.model.generate_dataset(ind_task - 1, nb_sample_test, one_task=False, Train=True)

            self.train_loader.concatenate(task_tr_gen)
            train_loader = self.train_loader[ind_task]
            train_loader.shuffle_task()

            if task_te_gen is not None:
                self.test_loader.concatenate(task_te_gen)
                test_loader = self.test_loader[ind_task]
                test_loader.shuffle_task()
            else:
                test_loader = None #we don't use test loader for instance but we keep the code for later in case of


        else:
            train_loader = self.train_loader[ind_task]
            test_loader = self.test_loader[ind_task]

        return train_loader, test_loader
