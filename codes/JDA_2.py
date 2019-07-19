# encoding=utf-8
"""
    Created on 9:38 2019/07/16
    @author: Chenxi Huang
    It implements "Transfer Feature Learning with Joint Distribution Adaptation"
    Refer to Wang Jindong's code in python
"""
import numpy as np
import os
import scipy.io
import scipy.linalg
import sklearn.metrics
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
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T  # compared to TCA, it add the iteration number

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        """
        list_acc = []
        # set predefined variables
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        # construct MMD matrix
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None

        for t in range(self.T):
            N = 0
            M0 = e * e.T * C# construct MMD matrix
            # the difference between TCA and JDA
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:  # Repeat
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])  # can't write tt is True. It is different.
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)

            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)

            n_eye = m if self.kernel_type == 'primal' else n
            # (X \sum_{c=0}^CM_c X^T + \lambda I)A=X H X^T A \Phi
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']   # databases: Office-31
    for i in range(1):
        for j in range(2):
            if i != j:
                src, tar = 'data/Office-31/' + domains[i], 'data/Office-31/' + domains[j]
                print("JDA:  data = Office-31 lambda = 1 src: " + src + " tar: " + tar)
                # load algorithm options
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                jda = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
                # because in office-31 all are objects, so lambda = 1
                acc, ypre, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
