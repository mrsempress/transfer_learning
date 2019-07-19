# encoding=utf-8
"""
    Created on 20:51 2019/07/14
    @author: Chenxi Huang
    It implements "Domain Adaptation via Transfer Component Analysis".
    Refer to Wang Jindong's code in python
"""
import numpy as np
import os
import scipy.io
import scipy.linalg  # scipy.linalg more quickly then numpy, and have more function, linalg = linear + algebra
import sklearn.metrics  # sklearn: scikit learn
from sklearn.neighbors import KNeighborsClassifier


def kernel(ker, X1, X2, gamma):
    """
    get the kernel
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
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':  # RBF: Radial Basis Function, and it use the idea of multivariate interpolation
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        """
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        """
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        """
        X = np.hstack((Xs.T, Xt.T))  # np.hstack(): tiling in the horizontal direction -> [Xs^T, Xt^T]
        # print(type(X))
        # print()
        # print(type(np.linalg.norm(X, axis=0)))
        X /= np.linalg.norm(X, axis=0)  # np.linalg.norm(X, axis=0), find X's form according to the columns
        m, n = X.shape  # n = ns + nt
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))  # (ns+nt, 1)

        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')  # frobenius=sqrt(\lambda(A^TA))
        H = np.eye(n) - 1 / n * np.ones((n, n))  # H=I_{ns+nt} - 1/(n_s+n_t)11^T, centering matrix
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)

        n_eye = m if self.kernel_type == 'primal' else n
        # a: KMK^T+\lambda Im   b: KHK^T
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)  # (KLK^T+\lambda I)^{-1}KHK^T, w is the eigenvalue, and v is eigenvectors
        ind = np.argsort(w)  # sort and return the index
        A = V[:, ind[:self.dim]]   # get the first m numbers
        Z = np.dot(A.T, K)  # W^TK
        Z /= np.linalg.norm(Z, axis=0)  # np.linalg.norm(Z, axis=0) = KWW^TK = the original K
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_red: accuracy and predicted labels on the target domain
        """
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)  # use K-Neighbors Classfier
        clf.fit(Xs_new, Ys.ravel())  # ravel: make it to one dimension
        y_pred = clf.predict(Xt_new)  # predict the value of Y
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)   # compare, and compute the accuracy
        return acc, y_pred


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # domains = ['amazon.mat', 'webcam.mat', 'dslr.mat']
    # name = ['amazonlist', 'webcamlist', 'dslrlist']

    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    # domains = ['caltech.mat', 'amazon.mat']
    for i in [2]:
        for j in [3]:
            if i != j:
                src, tar = 'data/Office-31/' + domains[i], 'data/Office-31/' + domains[j]
                print("TCA:  data = Office-31 lambda = 1 src: " + src + " tar: " + tar)
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
