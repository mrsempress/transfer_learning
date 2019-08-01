# encoding=utf-8
"""
    Created on 20:23 2019/07/24
    @author: Chenxi Huang
    It implement the paper "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation"
"""
from __future__ import print_function
from MCD_UDA import solver as S
import os
import torch


def work(source='svhn', target='mnist', gpu='3', max_epoch=200, resume_epoch=100, lr=0.0002,
         batch_size=128, optimizer='adam', num_k=4, all_use='no', checkpoint_dir='checkpoint', save_epoch=10,
         _save_model='False', _one_step='False', _use_abs_diff='False', _eval_only='False', seed=1):
    save_model = (_save_model == 'True')
    one_step = (_one_step == 'True')
    use_abs_diff = (_use_abs_diff == 'True')
    eval_only = (_eval_only == 'True')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    solver = S.Solver(use_abs_diff, eval_only, resume_epoch, source=source, target=target, learning_rate=lr,
                           batch_size=batch_size,
                           optimizer=optimizer, num_k=num_k, all_use=all_use,
                           checkpoint_dir=checkpoint_dir,
                           save_epoch=save_epoch)
    record_num = 0
    if source == 'usps' or target == 'usps':

        record_train = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s.txt' % (
            source, target, num_k, all_use, one_step, record_num)
        record_test = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s_test.txt' % (
            source, target, num_k, all_use, one_step, record_num)
        while os.path.exists(record_train):
            record_num += 1
            record_train = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s.txt' % (
                source, target, num_k, all_use, one_step, record_num)
            record_test = 'record/%s_%s_k_%s_alluse_%s_onestep_%s_%s_test.txt' % (
                source, target, num_k, all_use, one_step, record_num)
    else:
        record_train = 'record/%s_%s_k_%s_onestep_%s_%s.txt' % (
            source, target, num_k, one_step, record_num)
        record_test = 'record/%s_%s_k_%s_onestep_%s_%s_test.txt' % (
            source, target, num_k, one_step, record_num)
        while os.path.exists(record_train):
            record_num += 1
            record_train = 'record/%s_%s_k_%s_onestep_%s_%s.txt' % (
                source, target, num_k, one_step, record_num)
            record_test = 'record/%s_%s_k_%s_onestep_%s_%s_test.txt' % (
                source, target, num_k, one_step, record_num)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists('record'):
        os.mkdir('record')
    if eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(max_epoch):
            if not one_step:
                num = solver.train(t, record_file=record_train)
            else:
                num = solver.train_onestep(t, record_file=record_train)
            count += num
            if t % 1 == 0:
                solver.test(t, record_file=record_test, save_model=save_model)
            if count >= 20000:
                break


if __name__ == '__main__':
    work('svhn', 'mnist', num_k=4, _one_step='False')
