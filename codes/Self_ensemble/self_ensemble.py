# encoding=utf-8
"""
    Created on 15:05 2019/07/16
    @author: Chenxi Huang
    It implements "Self-ensemble Visual Domain Adapt Master" and refer to the writer's code
"""
import os
import time
import math
import numpy as np
import torch
from torch import nn
from batchup import data_source, work_pool
import Data_transform
import Network
from Self_ensemble import optim_weight_ema
from Self_ensemble import augmentation
from Self_ensemble import cmdline_helpers
from torch.nn import functional as F


# some parameters
# experiment to run: 'svhn_mnist', 'mnist_svhn', 'svhn_mnist_rgb', 'mnist_svhn_rgb', 'cifar_stl',
# 'stl_cifar', 'mnist_usps', 'usps_mnist', 'syndigits_svhn', 'synsigns_gtsrb'
exp = 'mnist_usps'
# network architecture: 'mnist-bn-32-64-256', 'grey-32-64-128-gp', 'grey-32-64-128-gp-wn',
# 'grey-32-64-128-gp-nonorm', 'rgb-128-256-down-gp', 'resnet18-32', 'rgb40-48-96-192-384-gp', 'rgb40-96-192-384-gp'
arch = ''
loss = 'var'  # augmentation variance loss function: 'var', 'bce'
double_softmax = False  # apply softmax twice to compute supervised loss
confidence_thresh = 0.96837722  # augmentation var loss confidence threshold !!! This is special.
rampup = 0  # ramp-up length
teacher_alpha = 0.99  # Teacher EMA alpha (decay)
fix_ema = False  # Use fixed EMA
unsup_weight = 3.0  # unsupervised loss weight
cls_bal_scale = False  # Enable scaling unsupervised loss to counteract class imbalance
cls_bal_scale_range = 0.0  # If not 0, clamp class imbalance scale to between x and 1/x where x is this value
cls_balance = 0.005  # Weight of class balancing component of unsupervised loss
cls_balance_loss = 'bce'  # Class balancing loss function: ['bce', 'log', 'bug']
combine_batches = False  # Build batches from both source and target samples
learning_rate = 0.001  # learning rate (Adam)
standardise_samples = False  # standardise samples (0 mean unit var)
src_hflip = False
src_xlat_range = 2.0
src_affine_std = 0.1
src_intens_flip = False
src_intens_scale_range = ''  # src aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)
src_intens_offset_range = ''  # src aug colour; intensity offset range `low:high` (-0.5:0.5 for mnist-svhn)
src_gaussian_noise_std = 0.1  # std aug: standard deviation of Gaussian noise to add to samples
tgt_affine_std = 0.1  # tgt aug xform: random affine transform std-dev
tgt_xlat_range = 2.0  # tgt aug xform: translation range
tgt_hflip = False  # tgt aug xform: enable random horizontal flips
tgt_intens_flip = False  # tgt aug colour; enable intensity flip
tgt_intens_scale_range = ''  # tgt aug colour; intensity scale range `low:high` (-1.5:1.5 for mnist-svhn)
tgt_intens_offset_range = ''  # tgt aug colour; intensity offset range `low:high` (-0.5:0.5 for mnist-svhn)
tgt_gaussian_noise_std = 0.1  # tgt aug: standard deviation of Gaussian noise to add to samples
num_epochs = 200  # number of epochs
batch_size = 64  # mini-batch size
epoch_size = 'target'  # 'large', 'small', 'target'
seed = 0  # random seed (0 for time-based)
log_file = ''  # log file path (none to disable
model_file = ''  # model file path
# device = 'cuda:3'  # Device
use_rampup = rampup > 0


def compute_aug_loss(stu_out, tea_out):
    # Augmentation loss
    if use_rampup:
        unsup_mask = None
        conf_mask_count = None
        unsup_mask_count = None
    else:
        conf_tea = torch.max(tea_out, 1)[0]
        # if smaller than confidence_thresh, than it as 0
        unsup_mask = conf_mask = (conf_tea > confidence_thresh).float()
        unsup_mask_count = conf_mask_count = conf_mask.sum()

    if loss == 'bce':
        aug_loss = Network.robust_binary_crossentropy(stu_out, tea_out)
    else:
        d_aug_loss = stu_out - tea_out
        aug_loss = d_aug_loss * d_aug_loss

    # Class balance scaling!!!
    if cls_bal_scale:
        if use_rampup:
            n_samples = float(aug_loss.shape[0])
        else:
            n_samples = unsup_mask.sum()
        avg_pred = n_samples / float(n_classes)
        bal_scale = avg_pred / torch.clamp(tea_out.sum(dim=0), min=1.0)
        if cls_bal_scale_range != 0.0:
            bal_scale = torch.clamp(bal_scale, min=1.0 / cls_bal_scale_range, max=cls_bal_scale_range)
        bal_scale = bal_scale.detach()
        aug_loss = aug_loss * bal_scale[None, :]

    aug_loss = aug_loss.mean(dim=1)

    if use_rampup:
        unsup_loss = aug_loss.mean() * rampup_weight_in_list[0]
    else:
        unsup_loss = (aug_loss * unsup_mask).mean()

    # Class balance loss
    if cls_balance > 0.0:
        # Compute per-sample average predicated probability
        # Average over samples to get average class prediction
        avg_cls_prob = stu_out.mean(dim=0)
        # Compute loss
        equalise_cls_loss = cls_bal_fn(avg_cls_prob, float(1.0 / n_classes))

        equalise_cls_loss = equalise_cls_loss.mean() * n_classes

        if use_rampup:
            equalise_cls_loss = equalise_cls_loss * rampup_weight_in_list[0]
        else:
            if rampup == 0:
                equalise_cls_loss = equalise_cls_loss * unsup_mask.mean(dim=0)

        unsup_loss += equalise_cls_loss * cls_balance

    return unsup_loss, conf_mask_count, unsup_mask_count


def f_train1(X_src0, X_src1, y_src, X_tgt0, X_tgt1):
    X_src0 = torch.tensor(X_src0, dtype=torch.float)
    X_src1 = torch.tensor(X_src1, dtype=torch.float)
    y_src = torch.tensor(y_src, dtype=torch.long)
    X_tgt0 = torch.tensor(X_tgt0, dtype=torch.float)
    X_tgt1 = torch.tensor(X_tgt1, dtype=torch.float)

    n_samples = X_src0.size()[0]
    n_total = n_samples + X_tgt0.size()[0]

    # student and teacher network to train
    student_optimizer.zero_grad()
    student_net.train()
    teacher_net.train()

    # Concatenate source and target mini-batches
    X0 = torch.cat([X_src0, X_tgt0], 0)
    X1 = torch.cat([X_src1, X_tgt1], 0)

    student_logits_out = student_net(X0)
    student_prob_out = F.softmax(student_logits_out, dim=1)

    src_logits_out = student_logits_out[:n_samples]
    src_prob_out = student_prob_out[:n_samples]

    teacher_logits_out = teacher_net(X1)
    teacher_prob_out = F.softmax(teacher_logits_out, dim=1)

    # Supervised classification loss
    if double_softmax:
        clf_loss = classification_criterion(src_prob_out, y_src)
    else:
        clf_loss = classification_criterion(src_logits_out, y_src)

    unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(student_prob_out, teacher_prob_out)

    loss_expr = clf_loss + unsup_loss * unsup_weight

    loss_expr.backward()
    student_optimizer.step()
    teacher_optimizer.step()

    outputs = [float(clf_loss) * n_samples, float(unsup_loss) * n_total]
    if not use_rampup:
        mask_count = float(conf_mask_count) * 0.5
        unsup_count = float(unsup_mask_count) * 0.5

        outputs.append(mask_count)
        outputs.append(unsup_count)
    return tuple(outputs)


def f_train2(X_src, y_src, X_tgt0, X_tgt1):
    X_src = torch.tensor(X_src, dtype=torch.float)
    y_src = torch.tensor(y_src, dtype=torch.long)
    X_tgt0 = torch.tensor(X_tgt0, dtype=torch.float)
    X_tgt1 = torch.tensor(X_tgt1, dtype=torch.float)

    # student and teacher network to train
    student_optimizer.zero_grad()
    student_net.train()
    teacher_net.train()

    src_logits_out = student_net(X_src)
    student_tgt_logits_out = student_net(X_tgt0)
    student_tgt_prob_out = F.softmax(student_tgt_logits_out, dim=1)
    teacher_tgt_logits_out = teacher_net(X_tgt1)
    teacher_tgt_prob_out = F.softmax(teacher_tgt_logits_out, dim=1)

    # Supervised classification loss
    if double_softmax:
        clf_loss = classification_criterion(F.softmax(src_logits_out, dim=1), y_src)
    else:
        clf_loss = classification_criterion(src_logits_out, y_src)

    unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(student_tgt_prob_out, teacher_tgt_prob_out)

    loss_expr = clf_loss + unsup_loss * unsup_weight

    loss_expr.backward()
    student_optimizer.step()
    teacher_optimizer.step()

    n_samples = X_src.size()[0]

    outputs = [float(clf_loss) * n_samples, float(unsup_loss) * n_samples]
    if not use_rampup:
        mask_count = float(conf_mask_count)
        unsup_count = float(unsup_mask_count)

        outputs.append(mask_count)
        outputs.append(unsup_count)
    return tuple(outputs)


def f_pred_src(X_sup):
    X_var = torch.tensor(X_sup, dtype=torch.float)
    student_net.eval()
    teacher_net.eval()
    return (F.softmax(student_net(X_var), dim=1).detach().cpu().numpy(),
            F.softmax(teacher_net(X_var), dim=1).detach().cpu().numpy())


def f_pred_tgt(X_sup):
    X_var = torch.tensor(X_sup, dtype=torch.float)
    student_net.eval()
    teacher_net.eval()
    return (F.softmax(student_net(X_var), dim=1).detach().cpu().numpy(),
            F.softmax(teacher_net(X_var), dim=1).detach().cpu().numpy())


def f_eval_src(X_sup, y_sup):
    y_pred_prob_stu, y_pred_prob_tea = f_pred_src(X_sup)
    y_pred_stu = np.argmax(y_pred_prob_stu, axis=1)
    y_pred_tea = np.argmax(y_pred_prob_tea, axis=1)
    return float((y_pred_stu != y_sup).sum()), float((y_pred_tea != y_sup).sum())


def f_eval_tgt(X_sup, y_sup):
    y_pred_prob_stu, y_pred_prob_tea = f_pred_tgt(X_sup)
    y_pred_stu = np.argmax(y_pred_prob_stu, axis=1)
    y_pred_tea = np.argmax(y_pred_prob_tea, axis=1)
    return float((y_pred_stu != y_sup).sum()), float((y_pred_tea != y_sup).sum())


if __name__ == '__main__':
    # choose the GPU and pool
    # torch_device = torch.device(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    pool = work_pool.WorkerThreadPool(2)

    src_intens_scale_range_lower, src_intens_scale_range_upper, src_intens_offset_range_lower, src_intens_offset_range_upper = \
        cmdline_helpers.intens_aug_options(src_intens_scale_range, src_intens_offset_range)
    tgt_intens_scale_range_lower, tgt_intens_scale_range_upper, tgt_intens_offset_range_lower, tgt_intens_offset_range_upper = \
        cmdline_helpers.intens_aug_options(tgt_intens_scale_range, tgt_intens_offset_range)

    # choose the source and target data
    if exp == 'svhn_mnist':
        d_source = Data_transform.load_svhn(zero_centre=False, greyscale=True)
        d_target = Data_transform.load_mnist(invert=False, zero_centre=False, pad32=True, val=False)
    elif exp == 'mnist_svhn':
        d_source = Data_transform.load_mnist(invert=False, zero_centre=False, pad32=True)
        d_target = Data_transform.load_svhn(zero_centre=False, greyscale=True, val=False)
    elif exp == 'svhn_mnist_rgb':
        d_source = Data_transform.load_svhn(zero_centre=False, greyscale=False)
        d_target = Data_transform.load_mnist(invert=False, zero_centre=False, pad32=True, val=False, rgb=True)
    elif exp == 'mnist_svhn_rgb':
        d_source = Data_transform.load_mnist(invert=False, zero_centre=False, pad32=True, rgb=True)
        d_target = Data_transform.load_svhn(zero_centre=False, greyscale=False, val=False)
    elif exp == 'cifar_stl':
        d_source = Data_transform.load_cifar10(range_01=False)
        d_target = Data_transform.load_stl(zero_centre=False, val=False)
    elif exp == 'stl_cifar':
        d_source = Data_transform.load_stl(zero_centre=False)
        d_target = Data_transform.load_cifar10(range_01=False, val=False)
    elif exp == 'mnist_usps':
        d_source = Data_transform.load_mnist(zero_centre=False)
        d_target = Data_transform.load_usps(zero_centre=False, scale28=True, val=False)
    elif exp == 'usps_mnist':
        d_source = Data_transform.load_usps(zero_centre=False, scale28=True)
        d_target = Data_transform.load_mnist(zero_centre=False, val=False)
    elif exp == 'syndigits_svhn':
        d_source = Data_transform.load_syn_digits(zero_centre=False)
        d_target = Data_transform.load_svhn(zero_centre=False, val=False)
    elif exp == 'synsigns_gtsrb':
        d_source = Data_transform.load_syn_signs(zero_centre=False)
        d_target = Data_transform.load_gtsrb(zero_centre=False, val=False)
    else:
        print('Unknown experiment type \'{}\''.format(exp))

    # standard it
    Data_transform.standardise_dataset(d_source)
    Data_transform.standardise_dataset(d_target)

    n_classes = d_source.n_classes

    print('===========Loaded data===========')
    if exp in {'mnist_usps', 'usps_mnist'}:
        arch = 'mnist-bn-32-64-256'
    if exp in {'svhn_mnist', 'mnist_svhn'}:
        arch = 'grey-32-64-128-gp'
    if exp in {'cifar_stl', 'stl_cifar', 'syndigits_svhn', 'svhn_mnist_rgb', 'mnist_svhn_rgb'}:
        arch = 'rgb-128-256-down-gp'
    if exp in {'synsigns_gtsrb'}:
        arch = 'rgb40-96-192-384-gp'

    net_class, expected_shape = Network.get_net_and_shape_for_architecture(arch)
    # if not the same shape, should give error and exit
    if expected_shape != d_source.train_X.shape[1:]:
        print('Architecture {} not compatible with experiment {}; it needs samples of shape {}, '
              'data has samples of shape {}'.format(arch, exp, expected_shape, d_source.train_X.shape[1:]))
        exit()

    student_net = net_class(n_classes)
    teacher_net = net_class(n_classes)
    student_params = list(student_net.parameters())
    teacher_params = list(teacher_net.parameters())
    for param in teacher_params:
        param.requires_grad = False

    # optimizer in student network and teacher network
    student_optimizer = torch.optim.Adam(student_params, lr=learning_rate)
    if fix_ema:
        teacher_optimizer = optim_weight_ema.EMAWeightOptimizer(teacher_net, student_net, alpha=teacher_alpha)
    else:
        teacher_optimizer = optim_weight_ema.OldWeightEMA(teacher_net, student_net, alpha=teacher_alpha)

    # Use the cross entropy loss
    classification_criterion = nn.CrossEntropyLoss()

    # show the data information
    print('Dataset:')
    print('SOURCE Train: X.shape={}, y.shape={}'.format(d_source.train_X.shape, d_source.train_y.shape))
    print('SOURCE Test: X.shape={}, y.shape={}'.format(d_source.test_X.shape, d_source.test_y.shape))
    print('TARGET Train: X.shape={}'.format(d_target.train_X.shape))
    print('TARGET Test: X.shape={}, y.shape={}'.format(d_target.test_X.shape, d_target.test_y.shape))

    print('===========Built network=========')
    src_aug = augmentation.ImageAugmentation(
        src_hflip, src_xlat_range, src_affine_std,
        intens_flip=src_intens_flip,
        intens_scale_range_lower=src_intens_scale_range_lower, intens_scale_range_upper=src_intens_scale_range_upper,
        intens_offset_range_lower=src_intens_offset_range_lower,
        intens_offset_range_upper=src_intens_offset_range_upper,
        gaussian_noise_std=src_gaussian_noise_std
    )
    tgt_aug = augmentation.ImageAugmentation(
        tgt_hflip, tgt_xlat_range, tgt_affine_std,
        intens_flip=tgt_intens_flip,
        intens_scale_range_lower=tgt_intens_scale_range_lower, intens_scale_range_upper=tgt_intens_scale_range_upper,
        intens_offset_range_lower=tgt_intens_offset_range_lower,
        intens_offset_range_upper=tgt_intens_offset_range_upper,
        gaussian_noise_std=tgt_gaussian_noise_std
    )

    if combine_batches:
        def augment(X_sup, y_src, X_tgt):
            X_src_stu, X_src_tea = src_aug.augment_pair(X_sup)
            X_tgt_stu, X_tgt_tea = tgt_aug.augment_pair(X_tgt)
            return X_src_stu, X_src_tea, y_src, X_tgt_stu, X_tgt_tea
    else:
        def augment(X_src, y_src, X_tgt):
            X_src = src_aug.augment(X_src)
            X_tgt_stu, X_tgt_tea = tgt_aug.augment_pair(X_tgt)
            return X_src, y_src, X_tgt_stu, X_tgt_tea

    rampup_weight_in_list = [0]

    cls_bal_fn = Network.get_cls_bal_function(cls_balance_loss)

    if combine_batches:
        f_train = f_train1
    else:
        f_train = f_train2

    print('===========Training==============')
    sup_ds = data_source.ArrayDataSource([d_source.train_X, d_source.train_y], repeats=-1)
    tgt_train_ds = data_source.ArrayDataSource([d_target.train_X], repeats=-1)
    train_ds = data_source.CompositeDataSource([sup_ds, tgt_train_ds]).map(augment)
    train_ds = pool.parallel_data_source(train_ds)

    if epoch_size == 'large':
        n_samples = max(d_source.train_X.shape[0], d_target.train_X.shape[0])
    elif epoch_size == 'small':
        n_samples = min(d_source.train_X.shape[0], d_target.train_X.shape[0])
    elif epoch_size == 'target':
        n_samples = d_target.train_X.shape[0]
    n_train_batches = n_samples // batch_size

    source_test_ds = data_source.ArrayDataSource([d_source.test_X, d_source.test_y])
    target_test_ds = data_source.ArrayDataSource([d_target.test_X, d_target.test_y])

    if seed != 0:
        shuffle_rng = np.random.RandomState(seed)
    else:
        shuffle_rng = np.random

    train_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)

    best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}

    best_conf_mask_rate = 0.0
    best_src_test_err = 1.0

    # Just doing it!
    for epoch in range(num_epochs):
        t1 = time.time()

        if use_rampup:
            if epoch < rampup:
                p = max(0.0, float(epoch)) / float(rampup)
                p = 1.0 - p
                rampup_value = math.exp(-p * p * 5.0)
            else:
                rampup_value = 1.0

            rampup_weight_in_list[0] = rampup_value

        train_res = data_source.batch_map_mean(f_train, train_batch_iter, n_batches=n_train_batches)

        train_clf_loss = train_res[0]
        if combine_batches:
            unsup_loss_string = 'unsup (both) loss={:.6f}'.format(train_res[1])
        else:
            unsup_loss_string = 'unsup (tgt) loss={:.6f}'.format(train_res[1])

        src_test_err_stu, src_test_err_tea = source_test_ds.batch_map_mean(f_eval_src, batch_size=batch_size * 2)
        tgt_test_err_stu, tgt_test_err_tea = target_test_ds.batch_map_mean(f_eval_tgt, batch_size=batch_size * 2)

        if use_rampup:
            unsup_loss_string = '{}, rampup={:.3%}'.format(unsup_loss_string, rampup_value)
            if src_test_err_stu < best_src_test_err:
                best_src_test_err = src_test_err_stu
                best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}
                improve = '*** '
            else:
                improve = ''
        else:
            conf_mask_rate = train_res[-2]
            unsup_mask_rate = train_res[-1]
            if conf_mask_rate > best_conf_mask_rate:
                best_conf_mask_rate = conf_mask_rate
                improve = '*** '
                best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}
            else:
                improve = ''
            unsup_loss_string = '{}, conf mask={:.3%}, unsup mask={:.3%}'.format(
                unsup_loss_string, conf_mask_rate, unsup_mask_rate)

        t2 = time.time()

        print('{}Epoch {} took {:.2f}s: \nTRAIN clf loss={:.6f}, {}; '
              'SRC TEST ERR={:.3%}, \nTGT TEST student err={:.3%}, TGT TEST teacher err={:.3%}'.format(
            improve, epoch, t2 - t1, train_clf_loss, unsup_loss_string, src_test_err_stu, tgt_test_err_stu,
            tgt_test_err_tea))

    # # Save network
    # if model_file != '':
    #     cmdline_helpers.ensure_containing_dir_exists(model_file)
    #     with open(model_file, 'wb') as f:
    #         pickle.dump(best_teacher_model_state, f)
