
from torchvision import transforms
from PIL import Image
import argparse
import os.path
import random
import torch

class Rotations(object):
    def __init__(self, args):
        super(Rotations, self).__init__()

        self.n_tasks = args.n_tasks
        self.i = args.i
        self.image_size = args.imageSize
        self.min_rot = args.min_rot
        self.max_rot = args.max_rot
        self.train_file = args.train_file
        self.test_file = args.test_file
        self.o = os.path.join(self.i, self.train_file).replace('training.pt', 'rotations_' + str(self.n_tasks) + '.pt')

        self.o_train = os.path.join(args.o, 'rotations_' + str(self.n_tasks) + '_train.pt')
        self.o_test = os.path.join(args.o, 'rotations_' + str(self.n_tasks) + '_test.pt')

    def rotate_dataset(self, d, rotation):
        result = torch.FloatTensor(d.size(0), 784)
        tensor = transforms.ToTensor()

        for i in range(d.size(0)):
            img = Image.fromarray(d[i].numpy(), mode='L')
            result[i] = tensor(img.rotate(rotation)).view(784)
        return result

    def formating_data(self):

        assert os.path.isfile(os.path.join(self.i, self.train_file))
        assert os.path.isfile(os.path.join(self.i, self.test_file))

        tasks_tr = []
        tasks_te = []

        x_tr, y_tr = torch.load(os.path.join(self.i, self.train_file))
        x_te, y_te = torch.load(os.path.join(self.i, self.test_file))

        for t in range(self.n_tasks):
            min_rot = 1.0 * t / self.n_tasks * (self.max_rot - self.min_rot) + \
                      self.min_rot
            max_rot = 1.0 * (t + 1) / self.n_tasks * \
                (self.max_rot - self.min_rot) + self.min_rot
            rot = random.random() * (max_rot - min_rot) + min_rot

            tasks_tr.append([rot, self.rotate_dataset(x_tr, rot), y_tr])
            tasks_te.append([rot, self.rotate_dataset(x_te, rot), y_te])

        torch.save(tasks_tr, self.o_train)
        torch.save(tasks_te, self.o_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--i', default='raw/', help='input directory')
    parser.add_argument('--o', default='mnist_rotations.pt', help='output file')
    parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
    parser.add_argument('--min_rot', default=0.,
                        type=float, help='minimum rotation')
    parser.add_argument('--max_rot', default=90.,
                        type=float, help='maximum rotation')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    DataFormatter = Mnist_Rotation(args)
    DataFormatter.formating_data()
