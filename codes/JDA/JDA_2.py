# encoding=utf-8
"""
    Created on 9:38 2019/07/16
    @author: Chenxi Huang
    It implements "Transfer Feature Learning with Joint Distribution Adaptation"
    Refer to Wang Jindong's code in python
"""
import Network
import os
import scipy.io
import Log
import Data_transform


def work(source, target, gpu, _lambd=1.0, _ker='primal', _gamma=1.0):
    # set log information
    log = Log.Log()
    log.set_dir('JDA', source, target)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']  # databases: Office-31

    Xs, Ys, Xt, Yt, A, B = None, None, None, None, None, None
    if source in ['amazon', 'caltech', 'webcam', 'dslr']:
        src, tar = 'data/Office-31/' + source + '.mat', 'data/Office-31/' + target + '.mat'
        print("JDA:  data = Office-31 lambda = 1 src: " + source + " tar: " + target)
        src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
        Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    # else:
    #     print("JDA:  data = Digits lambda = 1 src: " + source + " tar: " + target)
    #     if source == 'mnist':
    #         Xs, Ys, A, B = Data_transform.load_mnist_32()
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

    jda = Network.JDA(kernel_type=_ker, dim=30, lamb=_lambd, gamma=_gamma)
    # because in office-31 all are objects, so lambda = 1
    acc, ypre, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt, log)
    print(acc)

    # save log
    log.save_log()


if __name__ == '__main__':
    work('amazon', 'webcam', '3')
