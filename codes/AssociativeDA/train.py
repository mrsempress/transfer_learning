# encoding=utf-8
"""
    Created on 14:23 2019/07/24
    @author: Chenxi Huang
    It implements "Associative Domain Adaptation"
"""
import os
import os.path as osp
import torch
from AssociativeDA import solver
import Network
import Load_data


def work(source, target, gpu, sourcebatch=100, targetbatch=1000, checkpoint="", visit=0.1, walker=1.0, epochs=1000,
         learning_rate=3e-4, log="./log/ADA", num_workers=4):
    # Network
    if osp.exists(checkpoint):
        print("Resume from checkpoint file at {}".format(checkpoint))
        model = torch.load(checkpoint)
    else:
        model = Network.FrenchModel()

    # Adam optimizer, with amsgrad enabled
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999), amsgrad=True)

    # Dataset
    datasets = Load_data.load_dataset(path="data/Digist/", train=True)

    train_loader = torch.utils.data.DataLoader(datasets[source], batch_size=sourcebatch, shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(datasets[target], batch_size=targetbatch, shuffle=True,
                                             num_workers=num_workers)

    os.makedirs(log, exist_ok=True)
    solver.fit(model, optim, (train_loader, val_loader), n_epochs=epochs,
               savedir=log, visit_weight=visit, walker_weight=walker,
               cuda=int(gpu))


if __name__ == '__main__':
    work('mnist', 'svhn', '3')
