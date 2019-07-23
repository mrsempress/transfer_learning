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
import Network


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
            jda = Network.JDA_LMS(kernel_type=ker, dim=30, lamb=lambd, gamma=gamma)
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
