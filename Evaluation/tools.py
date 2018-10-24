import numpy as np
import torch
from scipy import linalg
from scipy.stats import entropy
#from sklearn.neighbors import KNeighborsClassifier
from torch.autograd import Variable

from log_utils import *

mpl.use('Agg')

import warnings





def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # token from https://github.com/bioinf-jku/TTUR/blob/master/fid.py

    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            #raise ValueError("Imaginary component {}".format(m))
            print('FID is fucked up')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def Inception_score(self):

    eval_size = 500

    # 0. load reference classifier
    self.load_best_baseline() #we load the best classifier

    # 1. generate data

    self.Classifier.eval()

    output_table = torch.Tensor(eval_size * self.batch_size, 10)

    # compute IS on real data
    if self.tau == 0:
        if len(self.test_loader) < eval_size:
            output_table = torch.Tensor((len(self.test_loader) - 1) * self.batch_size, 10)
        print("Computing of IS on test data")
        for i, (data, target) in enumerate(self.test_loader):
            if i >= eval_size or i >= (len(self.test_loader) - 1):  # (we throw away the last batch)
                break
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            label = Variable(target.squeeze())
            classif = self.Classifier(batch)
            output_table[i * self.batch_size:(i + 1) * self.batch_size, :] = classif.data
    elif self.tau == -1:
        if len(self.train_loader) < eval_size:
            output_table = torch.Tensor((len(self.train_loader) - 1) * self.batch_size, 10)
        print("Computing of IS on train data")
        for i, (data, target) in enumerate(self.train_loader):
            if i >= eval_size or i >= (len(self.train_loader) - 1):  # (we throw away the last batch)
                break
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            label = Variable(target.squeeze())
            classif = self.Classifier(batch)
            output_table[i * self.batch_size:(i + 1) * self.batch_size, :] = classif.data
    else:
        print("Computing of IS on generated data")
        for i in range(eval_size):
            data, target = self.generator.sample(self.batch_size)
            # 2. use the reference classifier to compute the output vector
            if self.gpu_mode:
                data, target = data.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            label = Variable(target.squeeze())
            classif = self.Classifier(batch)

            output_table[i * self.batch_size:(i + 1) * self.batch_size, :] = classif.data

    # Now compute the mean kl-div
    py = output_table.mean(0)

    assert py.shape[0] == 10

    scores = []
    for i in range(output_table.shape[0]):
        pyx = output_table[i, :]
        assert pyx.shape[0] == py.shape[0]
        scores.append(entropy(pyx.tolist(), py.tolist()))  # compute the KL-Divergence KL(P(Y|X)|P(Y))
    Inception_score = np.exp(np.asarray(scores).mean())

    if self.tau == 0:
        print("save reference IS")
        log_dir = os.path.join(self.log_dir, "..", "..", "..", "Classifier", 'seed_' + str(self.seed))
        np.savetxt(os.path.join(os.path.join(log_dir, 'Inception_score_ref_' + self.dataset + '.txt')),
                   np.transpose([Inception_score]))
    elif self.tau == -1:
        print("save IS evaluate on train")
        log_dir = os.path.join(self.log_dir, "..", "..", "..", "Classifier", 'seed_' + str(self.seed))
        np.savetxt(os.path.join(os.path.join(log_dir, 'Inception_score_train_' + self.dataset + '.txt')),
                   np.transpose([Inception_score]))
    else:
        np.savetxt(os.path.join(self.log_dir, 'Inception_score_' + self.dataset + '.txt'),
                   np.transpose([Inception_score]))

    print("Inception Score")
    print(Inception_score)


def knn(self):
    print("Training KNN Classifier")
    # Declare Classifier model
    data_samples = []
    label_samples = []

    # Training knn
    neigh = KNeighborsClassifier(n_neighbors=1)
    # We get the test data
    for i, (d, t) in enumerate(self.test_loader):
        if i == 0:
            data_test = d
            label_test = t
        else:
            data_test = torch.cat((data_test, d))
            label_test = torch.cat((label_test, t))
    data_test = data_test.numpy().reshape(-1, 784)
    label_test = label_test.numpy()
    # We get the training data
    for i, (d, t) in enumerate(self.train_loader):
        if i == 0:
            data_train = d
            label_train = t
        else:
            data_train = torch.cat((data_train, d))
            label_train = torch.cat((label_train, t))
    data = data_train.numpy().reshape(-1, 784)
    labels = label_train.numpy()

    if self.tau > 0:
        # we reduce the dataset
        data = data[0:int(len(data_train) * (1 - self.tau))]
        labels = labels[0:int(len(data_train) * (1 - self.tau))]
        # We get samples from the models
        for i in range(int((label_train.shape[0] * self.tau) / self.batch_size)):
            data_gen, label_gen = self.generator.sample(self.batch_size)
            data_samples.append(data_gen.cpu().numpy())
            label_samples.append(label_gen.cpu().numpy())

        # We concatenate training and gen samples
        data_samples = np.concatenate(data_samples).reshape(-1, 784)
        label_samples = np.concatenate(label_samples).squeeze()
        data = np.concatenate([data, data_samples])
        labels = np.concatenate([labels, label_samples])

    # We train knn
    neigh.fit(data, labels)
    accuracy = neigh.score(data_test,label_test)
    print("accuracy=%.2f%%" % (accuracy * 100))


    np.savetxt(os.path.join(self.log_dir, 'best_score_knn_' + self.dataset + '-tau' + str(self.tau) + '.txt'),
                   np.transpose([accuracy]))