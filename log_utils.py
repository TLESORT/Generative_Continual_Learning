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
import datetime


# first try for the log function it will be necessary to update it
def log_test_done(args, state='end'):
    f1 = open('test_done.txt', 'a')
    if args.context == "Generation":
        if args.train_G and state == 'Intermediate':
            f1.write('TrainG-{}-{}-{}-{}-{}-{}-{}\n'.format(args.seed, args.dataset, args.gan_type, args.method,
                                                            args.context, str(args.upperbound),
                                                            datetime.datetime.now()))
        elif args.eval or (args.train_G and state == 'End'):
            if args.FID:
                f1.write('FID-{}-{}-{}-{}-{}-{}\n'.format(args.seed, args.dataset, args.gan_type, args.method,
                                                              str(args.upperbound), datetime.datetime.now()))
            if args.Fitting_capacity:
                f1.write('Fitting_Capacity-{}-{}-{}-{}-{}-{}\n'.format(args.seed, args.dataset, args.gan_type, args.method,
                                                          str(args.upperbound), datetime.datetime.now()))
        else:
            f1.write('Log undefined -{}-{}-{}-{}-{}-{}\n'.format(args.seed, args.dataset, args.gan_type, args.method,
                                                                 str(args.upperbound), datetime.datetime.now()))
    elif args.context == "Classification":
        if args.eval:
            f1.write(
                'Classification-{}-{}-{}-{}-{}\n'.format(args.seed, args.dataset, args.method, str(args.upperbound),
                                                         datetime.datetime.now()))
        else:
            f1.write(
                'Log undefined2 -{}-{}-{}-{}-{}\n'.format(args.seed, args.dataset, args.method, str(args.upperbound),
                                                          datetime.datetime.now()))
    f1.close()


def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img) + 1e-12
    return img


def make_samples_batche(prediction, batch_size, filename_dest):
    plt.figure()
    batch_size_sqrt = int(np.sqrt(batch_size))
    input_channel = prediction[0].shape[0]
    input_dim = prediction[0].shape[1]
    prediction = np.clip(prediction, 0, 1)
    pred = np.rollaxis(prediction.reshape((batch_size_sqrt, batch_size_sqrt, input_channel, input_dim, input_dim)), 2,
                       5)
    pred = pred.swapaxes(2, 1)
    pred = pred.reshape((batch_size_sqrt * input_dim, batch_size_sqrt * input_dim, input_channel))
    fig, ax = plt.subplots(figsize=(batch_size_sqrt, batch_size_sqrt))
    ax.axis('off')
    ax.imshow(img_stretch(pred), interpolation='nearest')
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename_dest, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.close()


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0] / 255.0
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)


def loss_G_plot(hist, path='', model_name=''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()


def loss_plot(x, args):
    save_txt = np.zeros((len(x.items()), len(x[0])))
    for t, v in x.items():
        save_txt[t] = np.array(v)
        plt.plot(list(range(t * args.epochs, (t + 1) * args.epochs)), v)
        # plt.plot(list(range(t, (t + 1))), v)
    plt.savefig(os.path.join(args.log_dir, args.task_type + '_' + args.method + "_loss_figure.png"))
    plt.clf()
    np.savetxt(os.path.join(args.log_dir, args.task_type + '_' + args.method + "_loss.txt"), save_txt)


def accuracy_all_plot(x, args):
    save_txt = np.zeros((len(x.items()), len(x[0])))

    for t, v in x.items():
        save_txt[t, :len(v)] = np.array(v)

    plt.plot(range(len(x.items()) * len(x[0])), save_txt.reshape(len(x.items()) * len(x[0])))
    plt.savefig(os.path.join(args.log_dir, args.task_type + '_' + args.method + "_overall_accuracy_figure.png"))
    plt.clf()
    np.savetxt(os.path.join(args.log_dir, args.task_type + '_' + args.method + "_overall_accuracy.txt"), save_txt)

    max = save_txt.max()

    np.savetxt(os.path.join(args.log_dir, args.task_type + '_' + args.method + "_best_overall.txt"), [max])

def accuracy_plot(x, args):
    save_txt = np.ones((len(x.items()), len(x[0]))) * -1
    for t, v in x.items():
        save_txt[t, :len(v)] = np.array(v)
        plt.plot(list(range(t * args.epochs, args.num_task * args.epochs)), v)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(args.log_dir, args.task_type + '_' + args.method + "_accuracy_figure.png"))
    plt.clf()
    np.savetxt(os.path.join(args.log_dir, args.task_type + '_' + args.method + "_accuracy.txt"), save_txt)