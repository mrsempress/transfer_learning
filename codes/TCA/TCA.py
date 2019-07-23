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
                tca = Network.TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
