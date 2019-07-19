# encoding=utf-8
"""
    Created on 9:38 2019/07/16
    @author: Chenxi Huang
    It implements "Transfer Feature Learning with Joint Distribution Adaptation"
    Refer to Long Mingsheng's(the writer) code in Matlab
"""
import numpy as np
import os
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def kernel(ker, X1, X2, gamma):
    """
    get the Kernel
    :param ker: the type pf kernel
    :param X1: a domain
    :param X2: another domain
    :param gamma: use in the RBF model
    :return: the evaluate of pairwise distances or affinity of sets of samples
    """
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        """
        Init func
        :param kernel_type: kernel values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation, in paper is mu
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T  # compared to TCA, it add the iteration number

    def fit_predict(self, Xs, Ys, Xt, Yt0):
        """
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt0: nt * 1, target label
        :return: Z, A: the new input data and the first m eigenvalues
        """
        list_acc = []
        # set predefined variables
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape  # 800, 2081
        ns, nt = len(Xs), len(Xt)  # 1123, 958
        C = len(np.unique(Ys))
        # construct MMD matrix
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T * C
        if Yt0 is not None and len(Yt0) == nt:
            for c in np.reshape(np.unique(Ys), C, 1):
                e = np.zeros((n, n))
                e[np.where(Ys == c)] = 1 / len(Ys[np.where(Ys == c)])
                yy = Yt0 == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                e[tuple(inds)] = -1 / len(Yt0[np.where(Yt0 == c)])
                e[np.isinf(e)] = 0
                M = M + np.dot(e, e.T)
        M = M / np.linalg.norm(M, 'fro')

        # construct centering matrix
        H = np.eye(n) - 1 / n * np.ones((n, n))

        # Joint Distribution Adaptation: JDA
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        # (X \sum_{c=0}^CM_c X^T + \lambda I)A=X H X^T A \Phi
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        return Z, A


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']  # databases: Office-31
    srcStr = ['Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr',
              'dslr', 'dslr']
    tgtStr = ['amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10',
              'amazon', 'webcam']
    result = []
    for i in range(12):
        src, tar = './data/JDA/' + srcStr[i] + '_SURF_L10.mat', './data/JDA/' + tgtStr[i] + '_SURF_L10.mat'
        print("src is " + src + ", tar is " + tar)
        # load algorithm options
        src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
        # print(src_domain['fts'])
        # print(np.size(src_domain['fts'], 0))  # 1123
        # print(np.size(src_domain['fts'], 1))  # 800
        # print(src_domain['fts'].sum(0))
        # print(np.size(src_domain['fts'].sum(0), 0))  # 800
        # print(len(src_domain['fts']))  # 1123
        Xs = src_domain['fts'] / np.tile(src_domain['fts'].sum(0), 1)
        scale1 = preprocessing.minmax_scale(Xs, feature_range=(0, 1), axis=0, copy=True)
        # print(src_domain['labels'])
        Ys = src_domain['labels']

        Xt = tar_domain['fts'] / np.tile(tar_domain['fts'].sum(0), 1)
        scale2 = preprocessing.minmax_scale(Xs, feature_range=(0, 1), axis=0, copy=True)
        Yt = tar_domain['labels']

        # 1NN evaluation
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs, Ys.ravel())
        Y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, Y_pred)
        print('NN = ', acc)

        # JDA evaluation
        # because in office-31 all are objects, so lambda = 1
        k, lambd, ker, gamma = 100, 1.0, 'primal', 1.0  # 'primal' | 'linear' | 'rbf'
        T = 10
        Cls = []
        Acc = []
        for t in range(T):
            print('==============================Iteration [' + str(t) + ']==============================')
            jda = JDA(kernel_type=ker, dim=30, lamb=lambd, gamma=gamma)
            Z, A = jda.fit_predict(Xs, Ys, Xt, Yt)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :len(Xs)].T, Z[:, len(Xs):].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_pred = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_pred)
            Acc.append(acc)
            print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, T, acc))
        result.append(Acc[-1])
        print()
        print()
        print()
