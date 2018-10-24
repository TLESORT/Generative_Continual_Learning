import argparse
import os.path
import torch

class Permutations(object):
    def __init__(self, args):
        super(Permutations, self).__init__()

        self.n_tasks = args.n_tasks
        self.i = args.i
        self.train_file = args.train_file
        self.test_file = args.test_file

        self.o = os.path.join(self.i, self.train_file).replace('training.pt', 'permutations_' + str(self.n_tasks) + '.pt')
        #self.o = os.path.join(args.o, 'permutations_' + str(self.n_tasks) + '.pt')


        self.o_train = os.path.join(args.o, 'permutations_' + str(self.n_tasks) + '_train.pt')
        self.o_test = os.path.join(args.o, 'permutations_' + str(self.n_tasks) + '_test.pt')

    def formating_data(self):

        assert os.path.isfile(os.path.join(self.i, self.train_file))
        assert os.path.isfile(os.path.join(self.i, self.test_file))

        tasks_tr = []
        tasks_te = []

        x_tr, y_tr = torch.load(os.path.join(self.i, self.train_file))
        x_te, y_te = torch.load(os.path.join(self.i, self.test_file))
        x_tr = x_tr.float().view(x_tr.size(0), -1) / 255.0
        x_te = x_te.float().view(x_te.size(0), -1) / 255.0
        y_tr = y_tr.view(-1).long()
        y_te = y_te.view(-1).long()

        p = torch.FloatTensor(range(x_tr.size(1))).long()
        for t in range(self.n_tasks):

            tasks_tr.append(['random permutation', x_tr.index_select(1, p), y_tr])
            tasks_te.append(['random permutation', x_te.index_select(1, p), y_te])
            p = torch.randperm(x_tr.size(1)).long().view(-1)


        torch.save(tasks_tr, self.o_train)
        torch.save(tasks_te, self.o_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i', default='raw/cifar100.pt', help='input directory')
    parser.add_argument('--o', default='cifar100.pt', help='output file')
    parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--train_file', default='', help='input directory')
    parser.add_argument('--test_file', default='', help='input directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    DataFormatter = Permutations(args)
    DataFormatter.formating_data()
