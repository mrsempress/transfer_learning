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
                jda = Network.JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
                # because in office-31 all are objects, so lambda = 1
                acc, ypre, list_acc = jda.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
