
from Training.Trainer import Trainer


class Baseline(Trainer):
    def __init__(self, model, args, reviewer):
        super(Baseline, self).__init__(model, args, reviewer)

        self.reviewer = reviewer

    def preparation_4_task(self, model, ind_task):

        if ind_task > 0 and self.context == 'Generation':
            # We generate as much image as there is in the actual task
            #nb_sample_train = len(self.train_loader[ind_task])
            #nb_sample_test = int(nb_sample_train * 0.2)

            #print("nb_sample_train should be higher than this")
            print("numbe of train sample is fixed as : " + str(self.sample_transfer))
            nb_sample_train = self.sample_transfer
            nb_sample_test = int(nb_sample_train * 0.2)

            # we generate dataset for later evaluation with image from previous tasks
            self.model.generate_dataset(ind_task - 1, nb_sample_train, one_task=False, Train=True)
            #self.model.generate_dataset(ind_task - 1, nb_sample_test, one_task=False, Train=False)

            # generate dataset with one generator by task
            self.model.generate_best_dataset(ind_task - 1, nb_sample_train, Train=True)
            #self.model.generate_best_dataset(ind_task - 1, nb_sample_test, Train=False)

        train_loader, test_loader = self.create_next_data(ind_task)
        return train_loader, test_loader
