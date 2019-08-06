# encoding=utf-8
"""
    Created on 20:51 2019/07/14
    @author: Chenxi Huang
    It implements "Domain Adaptation via Transfer Component Analysis".
    Refer to Wang Jindong's code in python
"""
import os
import scipy.io
import scipy.linalg  # scipy.linalg more quickly then numpy, and have more function, linalg = linear + algebra
import Network
import Log
import Data_transform
import numpy as np


def work(source, target, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # domains = ['amazon.mat', 'webcam.mat', 'dslr.mat']
    # name = ['amazonlist', 'webcamlist', 'dslrlist']

    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']

    # set log information
    log = Log.Log()
    log.set_dir('TCA', source, target)

    Xs, Ys, Xt, Yt, A, B= None, None, None, None, None, None
    if source in ['amazon', 'caltech', 'webcam', 'dslr']:
        src, tar = 'data/Office-31/' + source + '.mat', 'data/Office-31/' + target + '.mat'
        print("TCA:  data = Office-31 lambda = 1 src: " + source + " tar: " + target)
        src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
        Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    # else:
    #     print("TCA:  data = Digits lambda = 1 src: " + source + " tar: " + target)
    #     if source == 'mnist':
    #         Xs, Ys, A, B= Data_transform.load_mnist_32()
    #     elif source == 'usps':
    #         Xs, Ys, A, B = Data_transform.load_usps_32()
    #     else:
    #         Xs, Ys, A, B = Data_transform.load_svhn_32()
    #
    #     if target == 'mnist':
    #         A, B, Xt, Yt = Data_transform.load_mnist_32()
    #     elif target == 'usps':
    #         A, B, Xt, Yt = Data_transform.load_usps_32()
    #     else:
    #         A, B, Xt, Yt = Data_transform.load_svhn_32()
    #     Ys = Ys.reshape(Ys.shape[0], 1)
    #     Yt = Yt.reshape(Yt.shape[0], 1)

    tca = Network.TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    acc, ypre, yacc = tca.fit_predict(Xs, Ys, Xt, Yt)
    print(acc)

    # add log and save
    log.add_log('*', '*', yacc, acc)
    log.save_log()


if __name__ == '__main__':
    work('amazon', 'webcam', '3')
