import argparse
import os

import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle

from copy import copy, deepcopy


def get_label(method, model, task):
    if method == 'Generative_Replay':
        predicat = "G_Replay"
    elif method == 'Baseline':
        predicat = "Fine-tuning"
    elif method == 'Baseline-up':
        predicat = "Up. Model"
    elif 'Rehearsal' in method:
        predicat = "Rehearsal"
    elif "Ewc" in method:
        predicat = method
    else:
        print(method + "<- Does not exist")

    if "upperbound_disjoint" in task:
        predicat = "Up. Data"


    if not model is None:
        label = predicat + '--' + model
    else:
        label = predicat

    return label


def load_best_score(path, num_task, dataset, method):
    list_best_score = []
    list_best_score_classes = []

    if method == "Baseline-up":
        id = 'Best_best_score_classif_' + dataset + '-task'
        id_classes = 'Best_data_classif_classes' + dataset + '-task'
    else:
        id = 'best_score_classif_' + dataset + '-task'
        id_classes = 'data_classif_classes' + dataset + '-task'

    for i in range(num_task):

        name_best_score = os.path.join(path, id + str(i) + '.txt')
        name_best_score_classes = os.path.join(path, id_classes + str(i) + '.txt')

        if os.path.isfile(name_best_score):
            best_score = np.array(np.loadtxt(name_best_score))
            list_best_score.append(best_score)
        else:
            print("Missing file : " + name_best_score)

        if os.path.isfile(name_best_score_classes):
            best_score_classes = np.array(np.loadtxt(name_best_score_classes))
            list_best_score_classes.append(best_score_classes)
        else:
            print("Missing file : " + name_best_score_classes)

    if not (len(list_best_score) == num_task and len(list_best_score_classes) == num_task):
        print("Missing file for : " + path)
        return [], []

    return np.array(list_best_score), np.array(list_best_score_classes)


def load_train_eval(path, num_task, dataset, method, conditional):

    if method == "Baseline-up":
        id = 'Best_'
    else:
        id = ''

    if conditional:
        file = 'TrainEval_All_Tasks'
        file_classes = 'TrainEval_classes_All_Tasks'

        name_best_score = os.path.join(path, id + file + '.txt')
        name_best_score_classes = os.path.join(path, id + file_classes + '.txt')

        if os.path.isfile(name_best_score):
            trainEval = np.array(np.loadtxt(name_best_score))
        else:
            print("Missing file : " + name_best_score)

        if os.path.isfile(name_best_score_classes):
            trainEval_classes = np.array(np.loadtxt(name_best_score_classes))
        else:
            print("Missing file : " + name_best_score_classes)

        return np.array(trainEval), np.array(trainEval_classes)

    else:
        return np.array(np.loadtxt(os.path.join(path, id + 'Balance_classes_All_Tasks.txt'))), None


def plot_perf_by_method_paper(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes, list_dataset,
                        Task):
    style_c = cycle(['-', '--', ':', '-.'])

    lign = len(list_method)
    column = len(list_dataset)

    for i, (Dataset) in enumerate(list_dataset):

        if Dataset == 'mnist':
            name_dataset = "MNIST"
        elif Dataset == 'fashion':
            name_dataset = "Fashion MNIST"

        for iter, (Method) in enumerate(list_method):

            # there is no results for this case
            if Task == 'upperbound_disjoint' and not Method == 'Baseline':
                continue

            for iter2, [best_result, dataset, method, model, num_task, task] in enumerate(
                    list_overall_best_score):

                if best_result.shape[0] == 0:
                    print("plot_perf_by_method : Problem with : " + str([dataset, method, model, num_task, task]))
                    print(best_result)
                    continue

                if method == Method and dataset == Dataset and task == Task:
                    label = model
                    if len(best_result) > 1:
                        best_result_mean = np.mean(best_result, axis=0)
                        best_result_std = np.std(best_result, axis=0)
                        plt.plot(range(num_task), best_result_mean, label=label, linestyle=next(style_c))
                        plt.fill_between(range(num_task), best_result_mean - best_result_std,
                                         best_result_mean + best_result_std, alpha=0.4)
                    else:
                        best_result = best_result.reshape(num_task)
                        plt.plot(range(num_task), best_result, label=label, linestyle=next(style_c))

            name_Method=get_label(Method, None, Task)
            plt.xlabel("Tasks")
            plt.ylim([0, 100])
            plt.title(name_Method)
            #plt.xticks([])

            #plt.yticks([])
            plt.ylabel('Fitting Capacity')

            plt.legend(loc=2, title='Algo', prop={'size': 6})
            # fig.text(0.04, 0.5, 'Fitting Capacity', va='center', ha='center', rotation='vertical')
            plt.ylabel("Fitting Capacity")
            plt.xticks(range(num_task))
            plt.xlabel(name_dataset + ' disjoint Tasks')
            plt.tight_layout()
            # fig.text(0.5, 0.04, 'Tasks', ha='center')
            plt.savefig(os.path.join(save_dir, Dataset + '_' + Task + '_' + Method + "_overall_accuracy.png"))
            plt.clf()


def plot_perf_by_method(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes, list_dataset,
                        Task):
    style_c = cycle(['-', '--', ':', '-.'])

    lign = len(list_method)
    column = len(list_dataset)

    fig, ax = plt.subplots(nrows=lign, ncols=column, sharex=True, sharey=True, figsize=(6, 8))

    for i, (Dataset) in enumerate(list_dataset):

        if Dataset == 'mnist':
            name_dataset = "MNIST"
        elif Dataset == 'fashion':
            name_dataset = "Fashion MNIST"

        for iter, (Method) in enumerate(list_method):
            indice = column * (iter) + i + 1
            plt.subplot(lign, column, indice)

            # there is no results for this case
            if Task == 'upperbound_disjoint' and not Method == 'Baseline':
                continue

            for iter2, [best_result, dataset, method, model, num_task, task] in enumerate(
                    list_overall_best_score):

                if best_result.shape[0] == 0:
                    print("plot_perf_by_method : Problem with : " + str([dataset, method, model, num_task, task]))
                    print(best_result)
                    continue

                if method == Method and dataset == Dataset and task == Task:
                    label = model
                    if len(best_result) > 1:
                        best_result_mean = np.mean(best_result, axis=0)
                        best_result_std = np.std(best_result, axis=0)
                        plt.plot(range(num_task), best_result_mean, label=label, linestyle=next(style_c))
                        plt.fill_between(range(num_task), best_result_mean - best_result_std,
                                         best_result_mean + best_result_std, alpha=0.4)
                    else:
                        best_result = best_result.reshape(num_task)
                        plt.plot(range(num_task), best_result, label=label, linestyle=next(style_c))

            name_Method=get_label(Method, None, Task)
            plt.xlabel("Tasks")
            plt.ylim([0, 100])
            plt.title(name_Method)
            plt.xticks([])

            plt.yticks([])
            plt.ylabel('Fitting Capacity')

            plt.legend(loc=2, title='Algo', prop={'size': 6})
            # fig.text(0.04, 0.5, 'Fitting Capacity', va='center', ha='center', rotation='vertical')
            plt.ylabel("Fitting Capacity")
        plt.xticks(range(num_task))
        plt.xlabel(name_dataset + ' disjoint Tasks')
    plt.tight_layout()
    # fig.text(0.5, 0.04, 'Tasks', ha='center')
    plt.savefig(os.path.join(save_dir, Task + "_overall_accuracy.png"))
    plt.clf()


def plot_perf_by_class(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes, Dataset, Task):
    style_c = cycle(['-', '--', ':', '-.'])

    for Method in list_method:

        # there is no results for this case
        if Task == 'upperbound_disjoint' and not Method == 'Baseline':
            continue

        for iter, [best_result_class, dataset, method, model, num_task, task] in enumerate(
                list_overall_best_score_classes):

            if method == Method and dataset == Dataset and task == Task:
                label = model

                if best_result_class.shape[0] == 0:
                    print("plot_perf_by_class : Problem with : " + str([dataset, method, model, num_task, task]))
                    print(best_result_class)
                    continue

                # print(best_result_class.shape)
                # [task, class]

                if len(best_result_class) > 1:
                    best_result_mean = np.mean(best_result_class, axis=0)
                    best_result_std = np.std(best_result_class, axis=0)
                    plt.plot(range(num_task), best_result_mean[:, 0], label=label, linestyle=next(style_c))
                    plt.fill_between(range(num_task), best_result_mean[:, 0] - best_result_std[:, 0],
                                     best_result_mean[:, 0] + best_result_std[:, 0], alpha=0.4)
                else:
                    best_result_class = best_result_class.reshape(num_task, 10)
                    plt.plot(range(num_task), best_result_class[:, 0], label=label, linestyle=next(style_c))

        plt.xlabel("Tasks")
        plt.ylabel("Task 0 Accuracy")
        plt.ylim([0, 100])
        plt.legend(loc=2, title='Algo')
        plt.title('accuracy_all_tasks')
        plt.savefig(os.path.join(save_dir, Dataset + '_' + Task + '_' + Method + "_task0_accuracy.png"))
        plt.clf()

def plot_variation(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes, Dataset, Task):
    style_c = cycle(['-', '--', ':', '-.'])
    classes = ["0", "1", "2", "3",
               "4", "5", "6", "7", "8", "9"]


    for Method in list_method:

        # there is no results for this case
        if Task == 'upperbound_disjoint' and not Method == 'Baseline':
            continue

        for iter, [best_result_class, dataset, method, model, num_task, task] in enumerate(
                list_overall_best_score_classes):

            tasks = range(num_task)

            fig, ax = plt.subplots()

            if method == Method and dataset == Dataset and task == Task:
                label = model
                # print(best_result_class.shape)
                # [seed, task, class]

                # if len(best_result_class) > 1:
                #     best_result_mean = np.mean(best_result_class, axis=0)
                #     grid = best_result_mean
                # else:
                #     grid = best_result_class

                # if grid.shape[0] == 0:
                #     print("plot_variation : Problem with : " + str([dataset, method, model, num_task, task]))
                #     continue
                #
                # grid = grid.reshape(10, num_task)

                #grid = grid.astype(int).transpose()
                #im = ax.imshow(grid)

                if len(best_result_class) > 1:



                    best_result_mean = np.mean(best_result_class, axis=0)
                    best_result_std = np.std(best_result_class, axis=0)
                    plt.plot(range(num_task), best_result_mean[:, 0], label=label, linestyle=next(style_c))
                    plt.fill_between(range(num_task), best_result_mean[:, 0] - best_result_std[:, 0],
                                     best_result_mean[:, 0] + best_result_std[:, 0], alpha=0.4)
                else:
                    best_result_class = best_result_class.reshape(num_task, 10)
                    plt.plot(range(num_task), best_result_class[:, 0], label=label, linestyle=next(style_c))


                plt.xlabel("Tasks")
                plt.ylabel("Mean Accuracy per Classes")
                plt.legend(loc=2, title='Algo')
                plt.title('Fitting Capacity Grid')
                plt.savefig(
                    os.path.join(save_dir, "models", Dataset + '_' + Task + '_' + Method + '_' + model + "_forgetting.png"))
                plt.clf()

def plot_grid_variations(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes, Dataset, Task):

    classes = ["0", "1", "2", "3",
               "4", "5", "6", "7", "8", "9"]


    for Method in list_method:

        # there is no results for this case
        if Task == 'upperbound_disjoint' and not Method == 'Baseline':
            continue

        for iter, [best_result_class, dataset, method, model, num_task, task] in enumerate(
                list_overall_best_score_classes):

            tasks = range(num_task)

            fig, ax = plt.subplots()

            if method == Method and dataset == Dataset and task == Task:
                # [seed, task, class]
                variation = deepcopy(best_result_class)

                for i in range(1, best_result_class.shape[1]): # from task 1 to last task (without task 0)
                    variation[:,i,:] -= best_result_class[:,i-1,:]

                if len(best_result_class) > 1:
                    variation_mean = np.mean(variation, axis=0)
                    grid = variation_mean
                else:
                    grid = variation

                if grid.shape[0] == 0:
                    print("plot_grid_class : Problem with : " + str([dataset, method, model, num_task, task]))
                    continue

                grid = grid.reshape(10, num_task)

                grid = grid.astype(int).transpose()
                im = ax.imshow(grid)

                # Create colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Fitting Capacity", rotation=-90, va="bottom")

                ax.set_xticks(np.arange(grid.shape[0]))  # task
                ax.set_yticks(np.arange(grid.shape[1]))  # classes
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                for i in range(len(tasks)):
                    for j in range(len(classes)):
                        text = ax.text(j, i, grid[i, j],
                                       ha="center", va="center", color="w")
                plt.xlabel("Tasks")
                plt.ylabel("Mean Variation Accuracy per Classes")
                plt.legend(loc=2, title='Algo')
                plt.title('Fitting Capacity Variation')
                plt.savefig(
                    os.path.join(save_dir, "models", Dataset + '_' + Task + '_' + Method + '_' + model + "_grid_variation.png"))
                plt.clf()

def plot_grid_class(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes, Dataset, Task):

    classes = ["0", "1", "2", "3",
               "4", "5", "6", "7", "8", "9"]


    for Method in list_method:

        # there is no results for this case
        if Task == 'upperbound_disjoint' and not Method == 'Baseline':
            continue

        for iter, [best_result_class, dataset, method, model, num_task, task] in enumerate(
                list_overall_best_score_classes):

            tasks = range(num_task)

            fig, ax = plt.subplots()

            if method == Method and dataset == Dataset and task == Task:

                # print(best_result_class.shape)
                # [seed, task, class]

                if len(best_result_class) > 1:
                    best_result_mean = np.mean(best_result_class, axis=0)
                    grid = best_result_mean
                else:
                    grid = best_result_class

                if grid.shape[0] == 0:
                    print("plot_grid_class : Problem with : " + str([dataset, method, model, num_task, task]))
                    continue

                grid = grid.reshape(10, num_task)

                grid = grid.astype(int).transpose()
                im = ax.imshow(grid)

                # Create colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

                ax.set_xticks(np.arange(grid.shape[0]))  # task
                ax.set_yticks(np.arange(grid.shape[1]))  # classes
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                for i in range(len(tasks)):
                    for j in range(len(classes)):
                        text = ax.text(j, i, grid[i, j],
                                       ha="center", va="center", color="w")
                plt.xlabel("Tasks")
                plt.ylabel("Mean Accuracy per Classes")
                plt.legend(loc=2, title='Algo')
                plt.title('Fitting Capacity Grid')
                plt.savefig(
                    os.path.join(save_dir, "models", Dataset + '_' + Task + '_' + Method + '_' + model + "_grid.png"))
                plt.clf()


def plot_FID(save_dir, list_model, list_FID, Dataset, Task):
    for Model in list_model:
        style_c = cycle(['-', '--', ':', '-.'])
        for iter, [FID_Values, dataset, method, model, num_task, task] in enumerate(
                list_FID):

            if model == Model and dataset == Dataset and (task == Task or task == "upperbound_" + Task):
                label = get_label(method, None, task)
                if len(FID_Values):
                    fid_mean = np.mean(FID_Values, axis=0)
                    fid_std = np.std(FID_Values, axis=0)
                    plt.plot(range(num_task), fid_mean, label=label, linestyle=next(style_c))
                    plt.fill_between(range(num_task), fid_mean - fid_std, fid_mean + fid_std, alpha=0.4)
                else:
                    FID_Values = FID_Values.reshape(num_task)
                    plt.plot(range(num_task), FID_Values, label=label, linestyle=next(style_c))

        plt.xlabel("Tasks")
        plt.ylabel("FID")
        plt.legend(loc=2, title='Algo')
        plt.savefig(os.path.join(save_dir, "models", Dataset + '_' + Task + '_' + Model + "_FID.png"))
        plt.clf()



def plot_models_perf(save_dir, list_model, list_overall_best_score, list_overall_best_score_classes, Dataset, Task):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for Model in list_model:
        style_c = cycle(['-', '--', ':', '-.'])
        for iter, [best_result, dataset, method, model, num_task, task] in enumerate(
                list_overall_best_score):

            if best_result.shape[0] == 0:
                print("plot_models_perf : Problem with : " + str([dataset, method, model, num_task, task]))
                continue

            if model == Model and dataset == Dataset and (task == Task or task == "upperbound_" + Task):
                label = get_label(method, None, task)
                if len(best_result) > 1:
                    best_result_mean = np.mean(best_result, axis=0)
                    best_result_std = np.std(best_result, axis=0)
                    plt.plot(range(num_task), best_result_mean, label=label, linestyle=next(style_c))
                    plt.fill_between(range(num_task), best_result_mean - best_result_std,
                                     best_result_mean + best_result_std, alpha=0.4)
                else:
                    best_result = best_result.reshape(num_task)
                    plt.plot(range(num_task), best_result, label=label, linestyle=next(style_c))

        plt.xlabel("Tasks")
        plt.ylabel("Fitting Capacity")
        plt.ylim([0, 100])
        plt.legend(loc=2, title='Algo')
        plt.title('FC_all_tasks')
        plt.savefig(os.path.join(save_dir, Dataset + '_' + Task + '_' + Model + "_overall_accuracy.png"))
        plt.clf()


def plot_FID_by_method(save_dir, list_method, list_FID, list_dataset, Task):
    style_c = cycle(['-', '--', ':', '-.'])

    lign = len(list_method)
    column = len(list_dataset)

    fig, ax = plt.subplots(nrows=lign, ncols=column, sharex=True, sharey=True, figsize=(8, 8))

    for i, (Dataset) in enumerate(list_dataset):

        for iter, (Method) in enumerate(list_method):
            indice = column * (iter) + i + 1
            plt.subplot(2, 2, indice)

            for iter2, [FID_Values, dataset, method, model, num_task, task] in enumerate(
                    list_FID):

                if method == Method and dataset == Dataset and (task == Task or task == "upperbound_" + Task):
                    label = model

                    if len(FID_Values) > 1:
                        fid_mean = np.mean(FID_Values, axis=0)
                        fid_std = np.std(FID_Values, axis=0)
                        plt.plot(range(num_task), fid_mean, label=label, linestyle=next(style_c))
                        plt.fill_between(range(num_task), fid_mean - fid_std, fid_mean + fid_std, alpha=0.4)
                    else:
                        FID_Values = FID_Values.reshape(num_task)
                        plt.plot(range(num_task), FID_Values, label=label, linestyle=next(style_c))

            plt.xticks(range(num_task))
            plt.title(Method)
            plt.legend(loc=0, title='Algo', prop={'size': 6})
            plt.ylabel("FID")


            plt.xlabel(Dataset + ' disjoint Tasks')

    fig.ylabel("FID")
    fig.text(0.04, 0.5, 'FID', va='center', ha='center', rotation='vertical')
    plt.tight_layout()
    fig.text(0.5, 0.04, 'Tasks', ha='center')
    plt.savefig(os.path.join(save_dir, Task + "_FID.png"))
    plt.clf()


def get_classifier_perf(folder, list_num_samples, dataset):
    list_best_score = []
    list_best_score_classes = []

    for num_samples in list_num_samples:

        if num_samples == 50000:

            # This should be simplified, latter only the file "disjoint_Baseline_best_overall" should be used

            if os.path.isfile(os.path.join(folder, "disjoint_Baseline_best_overall.txt")):
                best_score = np.loadtxt(os.path.join(folder, "disjoint_Baseline_best_overall.txt"))
            elif os.path.isfile(os.path.join(folder, "disjoint_Baseline_overall_accuracy.txt")):  # new name of the file
                best_score = np.max(np.loadtxt(os.path.join(folder, "disjoint_Baseline_overall_accuracy.txt")))
            elif os.path.isfile(
                    os.path.join(folder, "disjoint_Baseline_all_task_accuracy.txt")):  # old name of the file
                best_score = np.max(np.loadtxt(os.path.join(folder, "disjoint_Baseline_all_task_accuracy.txt")))
            else:
                print("No value for : " + os.path.join(folder, "disjoint_Baseline_best_overall.txt"))
            list_best_score.append(best_score)

        else:
            filename = "best_score_classif_" + str(dataset) + "-num_samples_" + str(num_samples) + ".txt"
            filename_classes = "data_classif_classes" + str(dataset) + "-num_samples_" + str(num_samples) + ".txt"

            if os.path.isfile(os.path.join(folder, filename)):
                best_score = np.loadtxt(os.path.join(folder, filename))
                list_best_score.append(best_score)
            else:
                print(os.path.join(folder, filename) + " : is missing ")

            if os.path.isfile(os.path.join(folder, filename_classes)):
                best_score_classes = np.loadtxt(os.path.join(folder, filename_classes))
                list_best_score_classes.append(best_score_classes)
            else:
                print(os.path.join(folder, filename_classes) + " : is missing ")

    return np.array(list_best_score), np.array(list_best_score_classes)


def plot_classif_perf(list_overall_best_score_classif, list_overall_best_score_classes_classif, list_num_samples,
                      Dataset):
    for iter, [scores_classif, dataset, method, num_task] in enumerate(list_overall_best_score_classif):

        if dataset == Dataset and num_task == 1 and method == "Baseline":

            scores_mean = scores_classif.mean(0)
            scores_std = scores_classif.std(0)

            # there should be only one curve by dataset
            plt.plot(list_num_samples, scores_mean)
            plt.fill_between(list_num_samples, scores_mean - scores_std, scores_mean + scores_std, alpha=0.4)

    plt.xscale('log')
    plt.xlabel("Number of Samples")
    plt.ylabel("Accuracy")
    plt.ylim([0, 100])
    plt.title('Accuracy in fonction number of samples used')
    plt.savefig(os.path.join(save_dir, Dataset + "_Accuracy_NbSamples.png"))
    plt.clf()


def print_perf(list_result_seed, dataset, method, model, num_task, task):
    #print('#############################################')
    # print(list_result_seed.shape) # (8, 10)

    if list_result_seed.shape[0] > 0:
        value = list_result_seed.max(0)
        print(dataset, method, model, num_task, task)
        print("Mean value :" + str(list_result_seed.mean(0)[-1]))
        print("std value :" + str(list_result_seed.std(0)[-1]))



def print_perf_trainEval(list_trainEval, dataset, method, model, num_task, task, conditional):
    if list_trainEval.shape[0] > 0:

        TrainEval = deepcopy(list_trainEval)
        if conditional:
            # print(TrainEval.shape) # (8, 10)
            value = TrainEval.max(0)[-1]
            print("TrainEval :" + str(value))
        else:
            # print(TrainEval.shape) # (8, 10, 10) # seed, task, classes # verifi'e

            np.set_printoptions(precision=2)
            np.set_printoptions(suppress=True)
            print(TrainEval[0, :, :])
            # iterate over seed
            TrainEval = deepcopy(list_trainEval)
            # iterate over seed
            for i in range(TrainEval.shape[0]):

                # iterate over tasks
                for j in range(TrainEval[i].shape[0]):
                    sum = TrainEval[i, j, :].sum()
                    if sum > 0:
                        TrainEval[i, j, :] = TrainEval[i, j, :] / sum
            print("Classes balanced")
            print(TrainEval.shape)
            print(TrainEval.max(0)[-1] * 100)


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--knn', type=bool, default=False)
    parser.add_argument('--IS', type=bool, default=False)
    parser.add_argument('--FID', type=bool, default=False)
    parser.add_argument('--BestPerf', type=bool, default=False)
    parser.add_argument('--others', type=bool, default=False)
    parser.add_argument('--trainEval', type=bool, default=False)
    parser.add_argument('--Accuracy', type=bool, default=False)
    parser.add_argument('--Diagram', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='logs', help='Logs directory')
    parser.add_argument('--save_dir', type=str, default='Figures_Paper', help='Figures directory')
    parser.add_argument('--comparatif', type=bool, default=False)
    parser.add_argument('--fitting_capacity', type=bool, default=False)
    parser.add_argument('--classif_perf', type=bool, default=False)
    parser.add_argument('--conditional', type=bool, default=False)

    return parser.parse_args()

log_dir = './Archives/logs_29_10'
save_dir = './Archives/Figures_29_10'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir + '/models')

args = parse_args()

liste_num = [100, 500, 1000, 5000, 10000, 50000]
liste_num = [50000]
list_seed = [0, 1, 2, 3, 4, 5, 6, 7]
# list_seed = [7]
list_val_tot = []
list_val_classes_tot = []

baseline_tot = []
baseline_classes_tot = []

baseline_all_seed = None
baseline_classes = None


list_model = ['GAN', 'CGAN', 'WGAN', 'VAE', 'CVAE', 'WGAN_GP']
list_method = ['Baseline', 'Baseline-up', 'Generative_Replay', 'Rehearsal', 'Ewc_5']
list_dataset = ['mnist', 'fashion']
list_task = ['disjoint', 'upperbound_disjoint']
# for evaluation of classifier only
list_num_samples = [10, 50, 100, 200, 500, 1000, 5000, 10000, 50000]

context = 'Generation'
# context = 'Classification'

list_num_task = [10]

list_generation_overall = []
list_generation_per_task = []

list_classification_overall = []
list_classification_per_task = []

list_overall_best_score = []
list_overall_best_score_classes = []

list_FID = []

#############################################   GET ALL INFO ######################################
if context == "Generation":
    for dataset in list_dataset:
        for task in list_task:
            for method in list_method:

                # there is no results for this case
                if task == 'upperbound_disjoint' and not method == 'Baseline':
                    continue

                for num_task in list_num_task:
                    for model in list_model:

                        if 'C' in model or method == 'Baseline-up':
                            args.conditional = True
                        else:
                            args.conditional = False

                        # we use list to group results for all seeds
                        list_result_seed = []
                        list_result_classes_seed = []
                        list_trainEval_seed = []
                        list_trainEval_classes_seed = []
                        list_result_FID_seed = []
                        for seed in list_seed:

                            if method == 'Baseline-up':
                                folder = os.path.join(log_dir, context, task, dataset, "Baseline", model,
                                                      "Num_tasks_" + str(num_task), "seed_" + str(seed))
                                file_FID = os.path.join(folder, "Best_Frechet_Inception_Distance_All_Tasks.txt")
                            else:
                                folder = os.path.join(log_dir, context, task, dataset, method, model,
                                                      "Num_tasks_" + str(num_task),
                                                      "seed_" + str(seed))
                                file_FID = os.path.join(folder, "Frechet_Inception_Distance_All_Tasks.txt")

                            overall_best_score, overall_best_score_classes = load_best_score(folder, num_task, dataset,
                                                                                             method)

                            if args.trainEval:
                                train_eval, train_eval_classes = load_train_eval(folder, num_task, dataset, method,
                                                                                 args.conditional)
                                list_trainEval_seed.append(train_eval)
                                list_trainEval_classes_seed.append(train_eval_classes)

                            if (overall_best_score is not None) and (overall_best_score != []):
                                list_result_seed.append(overall_best_score)
                            if (overall_best_score_classes is not None) and (overall_best_score_classes != []):
                                list_result_classes_seed.append(overall_best_score_classes)

                            if args.FID:
                                if os.path.isfile(file_FID):
                                    list_result_FID_seed.append(np.loadtxt(file_FID))
                                else:
                                    print("Missing FID File :  " + file_FID)

                        print_perf(np.array(list_result_seed), dataset, method, model, num_task, task)
                        if args.trainEval:
                            print_perf_trainEval(np.array(list_trainEval_seed), dataset, method, model, num_task, task,
                                                 args.conditional)

                        if list_result_seed != []:
                            list_overall_best_score.append(
                                [np.array(list_result_seed), dataset, method, model, num_task, task])

                        if list_result_classes_seed != []:
                            list_overall_best_score_classes.append(
                                [np.array(list_result_classes_seed), dataset, method, model, num_task, task])

                        if list_result_FID_seed != []:
                            list_FID.append([np.array(list_result_FID_seed), dataset, method, model, num_task, task])


elif context == "Classification":

    list_overall_best_score_classif = []
    list_overall_best_score_classes_classif = []

    for dataset in list_dataset:
        for method in list_method:
            for num_task in list_num_task:

                list_seed_score_classif = []
                list_seed_score_classif_classes = []

                for seed in list_seed:

                    folder = os.path.join(log_dir, context, dataset, method, "Num_tasks_" + str(num_task),
                                          "seed_" + str(seed))

                    if method == "Baseline" and num_task == 1:
                        overall_best_score, overall_best_score_classes = get_classifier_perf(folder, list_num_samples,
                                                                                             dataset)
                        list_seed_score_classif.append(overall_best_score)
                        list_seed_score_classif_classes.append(overall_best_score_classes)

                list_overall_best_score_classif.append(
                    [np.array(list_seed_score_classif), dataset, method, num_task])
                list_overall_best_score_classes_classif.append(
                    [np.array(list_seed_score_classif_classes), dataset, method, num_task])

if args.fitting_capacity:

    for dataset in list_dataset:
        for task in list_task:
            plot_grid_class(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes,
                            dataset, task)

            plot_grid_variations(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes,
                            dataset, task)

            if not 'upperbound' in task:
                plot_models_perf(save_dir + '/models', list_model, list_overall_best_score,
                                 list_overall_best_score_classes, dataset,
                                 task)
                plot_perf_by_class(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes,
                                    dataset, task)

    for task in list_task:
        plot_perf_by_method_paper(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes,
                            list_dataset, task)
        plot_perf_by_method(save_dir, list_method, list_overall_best_score, list_overall_best_score_classes,
                            list_dataset, task)

if args.FID:

    for task in list_task:
        plot_FID_by_method(save_dir, list_method, list_FID, list_dataset, task)


    for dataset in list_dataset:
        for task in list_task:
            if not 'upperbound' in task:
                plot_FID(save_dir, list_model, list_FID, dataset, task)

if args.classif_perf:

    for dataset in list_dataset:
        plot_classif_perf(list_overall_best_score_classif, list_overall_best_score_classes_classif, list_num_samples,
                          dataset)
