import argparse
import os

import numpy as np
import torch

from Classifiers.Fashion_Classifier import Fashion_Classifier
from Classifiers.Mnist_Classifier import Mnist_Classifier
from Classifiers.Cifar_Classifier import Cifar_Classifier
from Data.data_loader import DataLoader
from Evaluation.Reviewer import Reviewer
from Generative_Models.BEGAN import BEGAN
from Generative_Models.CGAN import CGAN
from Generative_Models.CVAE import CVAE
from Generative_Models.GAN import GAN
from Generative_Models.VAE import VAE
from Generative_Models.WGAN import WGAN
from Generative_Models.WGAN_GP import WGAN_GP
from Training.Baseline import Baseline
from Training.Ewc import Ewc
from Training.Ewc_samples import Ewc_samples
from Training.Generative_Replay import Generative_Replay
from Training.Rehearsal import Rehearsal
from log_utils import log_test_done
from utils import check_args

from Evaluation.Eval_Classifier import Reviewer_C

"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='CVAE',
                        choices=['GAN', 'Classifier', 'CGAN', 'BEGAN', 'WGAN',
                                 'WGAN_GP', 'VAE', "CVAE", "WGAN_GP"],
                        help='The type of GAN')  # , required=True)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion', 'cifar10'],
                        help='The name of dataset')
    parser.add_argument('--conditional', type=bool, default=False)
    parser.add_argument('--upperbound', type=bool, default=False,
                        help='This variable will be set to true automatically if task_type contains_upperbound')
    parser.add_argument('--method', type=str, default='Baseline', choices=['Baseline', 'Ewc', 'Ewc_samples',
                                                                           'Generative_Replay', 'Rehearsal'])

    parser.add_argument('--context', type=str, default='Generation',
                        choices=['Classification', 'Generation', 'Not_Incremental'])

    parser.add_argument('--dir', type=str, default='./Archives/', help='Working directory')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save results')
    parser.add_argument('--sample_dir', type=str, default='Samples', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--data_dir', type=str, default='Data', help='Directory name for data')
    parser.add_argument('--gen_dir', type=str, default='.', help='Directory name for data')

    parser.add_argument('--epochs', type=int, default=1, help='The number of epochs to run')
    parser.add_argument('--epoch_G', type=int, default=1, help='The number of epochs to run')
    parser.add_argument('--epoch_Review', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--size_epoch', type=int, default=1000)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--verbose', type=bool, default=False)

    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lrC', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--seed', type=int, default=1664)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--train_G', type=bool, default=False)
    parser.add_argument('--eval_C', type=bool, default=False)

    ############### UNUSED FLAGS ##########################
    parser.add_argument('--trainEval', type=bool, default=False)
    parser.add_argument('--knn', type=bool, default=False)
    parser.add_argument('--IS', type=bool, default=False)
    parser.add_argument('--FID', type=bool, default=False)
    parser.add_argument('--Fitting_capacity', type=bool, default=False)
    #######################################################

    parser.add_argument('--num_task', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--sample_transfer', type=int, default=5000)
    parser.add_argument('--task_type', type=str, default="disjoint",
                        choices=['disjoint', 'permutations', 'upperbound_disjoint'])
    parser.add_argument('--samples_per_task', type=int, default=200)
    parser.add_argument('--lambda_EWC', type=int, default=5)
    parser.add_argument('--nb_samples_reharsal', type=int, default=10)
    parser.add_argument('--regenerate', type=bool, default=False)

    return check_args(parser.parse_args())


"""main"""


def main():
    # parse arguments
    args = parse_args()

    if args is None:
        exit()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    args.gpu_mode = torch.cuda.is_available()

    if args.gpu_mode:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    if args.context == 'Generation':
        print("Generation : Use of model {} with dataset {}, seed={}".format(args.gan_type, args.dataset, args.seed))
    elif args.context == 'Classification':
        print("Classification : Use of method {} with dataset {}, seed={}".format(args.method, args.dataset, args.seed))

    if args.context == 'Generation':

        # declare instance for GAN
        if args.gan_type == 'GAN':
            model = GAN(args)
        elif args.gan_type == 'CGAN':
            model = CGAN(args)
        elif args.gan_type == 'VAE':
            model = VAE(args)
        elif args.gan_type == 'CVAE':
            model = CVAE(args)
        elif args.gan_type == 'WGAN':
            model = WGAN(args)
        elif args.gan_type == 'WGAN_GP':
            model = WGAN_GP(args)
        else:
            raise Exception("[!] There is no option for " + args.gan_type)
    elif args.context == 'Classification':
        if args.dataset == 'mnist':
            model = Mnist_Classifier(args)
        elif args.dataset == 'fashion':
            model = Fashion_Classifier(args)
        elif args.dataset == 'cifar10':
            model = Cifar_Classifier(args)
        else:
            print('Not implemented')

    reviewer = Reviewer(args)

    if args.method == 'Baseline':
        method = Baseline(model, args, reviewer)
    elif 'Ewc' in args.method:
        method = Ewc(model, args)
    elif 'Ewc_samples' in args.method:
        method = Ewc_samples(model, args)
    elif args.method == 'Generative_Replay':
        method = Generative_Replay(model, args)
    elif args.method == 'Rehearsal':
        method = Rehearsal(model, args)
    else:
        print('Method not implemented')

    if args.context == 'Classification':

        if args.eval_C:
            reviewer_C = Reviewer_C(args)
            list_values = [10, 50, 100, 200, 500, 1000, 5000, 10000]

            reviewer_C.review_all_tasks(args, list_values)
        else:
            method.run_classification_tasks()
    elif args.context == 'Generation':

        if args.train_G:
            method.run_generation_tasks()
            log_test_done(args, 'Intermediate')

        if args.regenerate:
            method.regenerate_datasets_for_eval()

        if args.Fitting_capacity and not args.train_G:

            reviewer = Reviewer(args)
            # In case the training training and evaluation are done separately
            if args.gpu_mode:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed_all(args.seed)

            reviewer.review_all_tasks(args)

            if args.method == "Baseline" and not args.upperbound:
                # Baseline produce both one lower bound and one upperbound
                # it is not the same upperbound a the one trained for upperbound_disjoint
                reviewer.review_all_tasks(args, Best=True)

        if args.FID and not args.train_G:

            reviewer = Reviewer(args)
            # In case the training training and evaluation are done separately
            if args.gpu_mode:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed_all(args.seed)

            reviewer.compute_all_tasks_FID(args)

            if args.method == "Baseline" and not args.upperbound:
                # Baseline produce both one lower bound and one upperbound
                # it is not the same upperbound a the one trained for upperbound_disjoint
                reviewer.compute_all_tasks_FID(args, Best=True)

        if args.trainEval:
            reviewer = Reviewer(args)
            reviewer.review_all_trainEval(args)

            if args.method == "Baseline" and not args.upperbound:
                # Baseline produce both one lower bound and one upperbound
                # it is not the same upperbound a the one trained for upperbound_disjoint
                reviewer.review_all_trainEval(args, Best=True)


    else:
        print('Not Implemented')

    log_test_done(args)


if __name__ == '__main__':
    main()
