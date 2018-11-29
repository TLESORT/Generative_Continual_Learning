import torch
import numpy as np
from log_utils import *

class DataLoader(object):
    def __init__(self, data, args):

        '''

        dataset.shape = [num , 3, image_number]
        dataset[0 , 1, :] # all data from task 0
        dataset[0 , 2, :] # all label from task 0

        '''

        self.dataset = data
        self.batch_size = args.batch_size
        n_tasks = args.num_task
        self.length = n_tasks
        self.current_sample = 0
        self.current_task = 0
        self.sampler = self

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        '''

        :return: (data, label) with shape batch_size

        '''

        if self.current_sample == self.dataset[self.current_task][1].shape[0]:
            self.current_sample = 0 # reinitialize
            self.shuffle_task()
            raise StopIteration
        elif self.current_sample + self.batch_size >= self.dataset[self.current_task][1].shape[0]:
            last_size = self.dataset[self.current_task][1].shape[0] - self.current_sample
            j = range(self.current_sample, self.current_sample + last_size)
            self.current_sample = self.current_sample + last_size
            j = torch.LongTensor(j)
            return self.dataset[self.current_task][1][j], self.dataset[self.current_task][2][j]
        else:
            j = range(self.current_sample, self.current_sample + self.batch_size)
            self.current_sample = self.current_sample + self.batch_size
            j = torch.LongTensor(j)
            return self.dataset[self.current_task][1][j], self.dataset[self.current_task][2][j]

    def __len__(self):
        return len(self.dataset[self.current_task][1])

    def __getitem__(self, key):
        self.current_sample = 0
        self.current_task = key
        return self


    def shuffle_task(self):
        indices = torch.randperm(len(self.dataset[self.current_task][1]))
        self.dataset[self.current_task][1] = self.dataset[self.current_task][1][indices].clone()
        self.dataset[self.current_task][2] = self.dataset[self.current_task][2][indices].clone()

    def get_sample(self, number):
        indices = torch.randperm(len(self))[0:number]

        return self.dataset[self.current_task][1][indices], self.dataset[self.current_task][2][indices]

    def concatenate(self, new_data, task=0):

        '''

        :param new_data: data to add to the actual task
        :return: the actual dataset with supplementary data inside
        '''

        self.dataset[self.current_task][1] = torch.cat((self.dataset[self.current_task][1], new_data.dataset[task][1]), 0).clone()
        self.dataset[self.current_task][2] = torch.cat((self.dataset[self.current_task][2], new_data.dataset[task][2]), 0).clone()

        return self

    def get_current_task(self):
        return self.current_task

    def save(self, path):
        torch.save(self.dataset, path)

    def visualize_sample(self, path , number, shape):
        data, target = self.get_sample(number)

        # get sample in order from 0 to 9
        target, order = target.sort()
        data = data[order]

        image_frame_dim = int(np.floor(np.sqrt(number)))

        if shape[2] == 1:
            data = data.numpy().reshape(number, shape[0], shape[1], shape[2])
            save_images(data[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        path)
        else:
            data = data.numpy().reshape(number, shape[2], shape[1], shape[0])
            make_samples_batche(data[:number], number, path)

    def increase_size(self, increase_factor):

        self.dataset[self.current_task][1] = torch.cat([self.dataset[self.current_task][1]]*increase_factor, 0)
        self.dataset[self.current_task][2] = torch.cat([self.dataset[self.current_task][2]]*increase_factor, 0)

        return self


