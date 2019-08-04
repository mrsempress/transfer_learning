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


def work(source, target, gpu, _lambd=1.0, _ker='primal', _gamma=1.0):
    # set log information
    log = Log.Log()
    log.set_dir('JDA', source, target)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']  # databases: Office-31

    src, tar = 'data/Office-31/' + source + '.mat', 'data/Office-31/' + target + '.mat'
    print("JDA:  data = Office-31 lambda = 1 src: " + src + " tar: " + tar)
    # load algorithm options
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    jda = Network.JDA(kernel_type=_ker, dim=30, lamb=_lambd, gamma=_gamma)
    # because in office-31 all are objects, so lambda = 1
    acc, ypre, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt, log)
    print(acc)

    # save log
    log.save_log()


if __name__ == '__main__':
    work('amazon', 'webcam', '3')
