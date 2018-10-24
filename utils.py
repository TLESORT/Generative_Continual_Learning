import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable

"""checking arguments"""


def check_args(args):
    if "Ewc" in args.method:
        args.method = args.method + '_' + str(args.lambda_EWC)

    args.save_dir = os.path.join(args.dir, args.save_dir)
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.result_dir = os.path.join(args.dir, args.result_dir)
    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.log_dir = os.path.join(args.dir, args.log_dir)
    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    args.sample_dir = os.path.join(args.dir, args.sample_dir)
    # --sample_dir
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    args.data_dir = os.path.join(args.dir, args.data_dir)
    # --sample_dir
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    args.gen_dir = os.path.join(args.data_dir, 'Generated')
    # --sample_dir
    if not os.path.exists(args.gen_dir):
        os.makedirs(args.gen_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    if 'upperbound' in args.task_type:
        args.upperbound = True
    elif (not 'upperbound' in args.task_type) and args.upperbound:
        args.task_type = 'upperbound_' + args.task_type

    args.data_file = args.task_type + '_' + str(args.num_task) + '.pt'

    #
    if args.gan_type == "VAE" and args.conditional:
        args.gan_type = "CVAE"
    if args.gan_type == "GAN" and args.conditional:
        args.gan_type = "CGAN"

    if args.context == 'Generation':
        args.result_dir = os.path.join(args.result_dir, args.context, args.task_type, args.dataset, args.method,
                                       args.gan_type,
                                       'Num_tasks_' + str(args.num_task),
                                       'seed_' + str(args.seed))
        args.save_dir = os.path.join(args.save_dir, args.context, args.task_type, args.dataset, args.method,
                                     args.gan_type,
                                     'Num_tasks_' + str(args.num_task), 'seed_' + str(args.seed))
        args.log_dir = os.path.join(args.log_dir, args.context, args.task_type, args.dataset, args.method,
                                    args.gan_type,
                                    'Num_tasks_' + str(args.num_task), 'seed_' + str(args.seed))
        args.sample_dir = os.path.join(args.sample_dir, args.context, args.task_type, args.dataset, args.gan_type,
                                       args.method,
                                       'Num_tasks_' + str(args.num_task),
                                       'seed_' + str(args.seed))
        args.gen_dir = os.path.join(args.gen_dir, args.dataset, args.gan_type, args.task_type, args.method,
                                    'Num_tasks_' + str(args.num_task),
                                    'seed_' + str(args.seed))

    elif args.context == 'Classification':
        args.result_dir = os.path.join(args.result_dir, args.context, args.dataset, args.method,
                                       'Num_tasks_' + str(args.num_task),
                                       'seed_' + str(args.seed))
        args.save_dir = os.path.join(args.save_dir, args.context, args.dataset, args.method,
                                     'Num_tasks_' + str(args.num_task), 'seed_' + str(args.seed))
        args.log_dir = os.path.join(args.log_dir, args.context, args.dataset, args.method,
                                    'Num_tasks_' + str(args.num_task), 'seed_' + str(args.seed))
        args.sample_dir = os.path.join(args.sample_dir, args.context, args.dataset, args.task_type,
                                       'Num_tasks_' + str(args.num_task),
                                       'seed_' + str(args.seed))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.gen_dir):
        os.makedirs(args.gen_dir)

    if args.gan_type == "CVAE" or args.gan_type == "CGAN":
        args.conditional = True

    print("Model     : ", args.gan_type)
    print("Dataset   : ", args.dataset)
    print("Method    : ", args.method)
    print("Seed      : ", str(args.seed))
    print("Context   : ", args.context)

    if args.FID:
        print("Doing     : FID")
    if args.train_G:
        print("Doing     : Train_G")
    if args.Fitting_capacity:
        print("Doing     : Fitting_capacity")

    return args


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def load_datasets(args):
    print(args.data_file)

    train_file = args.data_file.replace('.pt', '_train.pt')
    test_file = args.data_file.replace('.pt', '_test.pt')

    data_train = torch.load(os.path.join(args.data_dir, 'Tasks', args.dataset, train_file))
    data_test = torch.load(os.path.join(args.data_dir, 'Tasks', args.dataset, test_file))

    n_inputs = data_train[0][1].size(1)
    n_outputs = 0
    for i in range(len(data_train)):
        n_outputs = max(n_outputs, data_train[i][2].max())
        n_outputs = max(n_outputs, data_test[i][2].max())

    return data_train, data_test, n_inputs, n_outputs + 1, len(data_train)
