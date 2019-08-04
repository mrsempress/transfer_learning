# encoding=utf-8
"""
    Created on 14:23 2019/07/24
    @author: Chenxi Huang
    Minimal Training Script for Associative Domain Adaptation
"""
import time
import pandas as pd
import tqdm
import torch
from torch import nn
import Loss


def dual_iterator(ref, other):
    other_it = zip(other)
    for data in ref:
        data[0] = torch.cat([data[0], data[0], data[0]], dim=1)
        try:
            data_, = next(other_it)
        except StopIteration:
            other_it = zip(other)
            data_,   = next(other_it)

        yield data, data_


def fit(log, model, optim, dataset,
        n_epochs=10, walker_weight=1., visit_weight=.1, cuda=None):

    train, val = dataset

    cudafy = lambda x: x if cuda is None else x.cuda(cuda)
    torch2np = lambda x: x.cpu().detach().numpy()

    DA_loss = Loss.AssociativeLoss(walker_weight=walker_weight, visit_weight=visit_weight)
    CL_loss = nn.CrossEntropyLoss()

    cudafy(model)
    model.train()

    print('training start!')
    start_time = time.time()

    num_iter = 0
    train_hist = []
    pbar_epoch = tqdm.tqdm(range(n_epochs))
    tic = time.time()
    for epoch in pbar_epoch:
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()

        pbar_batch = tqdm.tqdm(dual_iterator(*dataset))
        accs = None
        acct = None
        for (xt, yt), (xs, ys) in pbar_batch:

            xs = cudafy(xs)
            ys = cudafy(ys)
            xt = cudafy(xt)
            yt = cudafy(yt)

            losses = {}

            # D CL training
            model.zero_grad()

            phi_s, yp = model(xs)
            phi_t, ypt = model(xt)

            yp = yp.squeeze().clone()
            ypt = ypt.squeeze().clone()

            losses['D class'] = CL_loss(yp, ys).mean()
            losses['D adapt'] = DA_loss(phi_s, phi_t, ys).mean()

            losses['D acc src'] = torch.eq(yp.max(dim=1)[1], ys).sum().float() / 200
            losses['D acc tgt'] = torch.eq(ypt.max(dim=1)[1], yt).sum().float() / 20

            (losses['D class'] + losses['D adapt']).backward()
            optim.step()

            losses = {k: v.cpu().data.detach().numpy() for k, v in losses.items()}
            losses['batch'] = num_iter
            train_hist.append(losses)

            num_iter += 1

            df = pd.DataFrame(train_hist)

            # df.to_csv(osp.join(savedir, 'losshistory.csv'))

            acc_s = df['D acc src'][-100:].mean()
            acc_t = df['D acc tgt'][-100:].mean()

            if accs == None:
                accs = acc_s
            else:
                accs = max(accs, acc_s)

            if acct == None:
                acct = acc_t
            else:
                acct = max(acct, acc_t)

        print('Epoch {} - S {:.3}% - T {:.3}%'.format(epoch, accs*100, acct*100))
        log.add_log(epoch, '*', accs, acct)
