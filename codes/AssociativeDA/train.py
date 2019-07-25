# encoding=utf-8
"""
    Created on 14:23 2019/07/24
    @author: Chenxi Huang
    It implements "Associative Domain Adaptation"
"""

import os
import os.path as osp
import argparse
import torch
from ADA import solver
import Network
import Load_data


def build_parser():

    parser = argparse.ArgumentParser(description='Associative Domain Adaptation')

    # General setup
    parser.add_argument('--gpu', default=0, help='Specify GPU', type=int)
    parser.add_argument('--cpu', action='store_true', help="Use CPU Training")
    parser.add_argument('--log', default="./log/log2", help="Log directory. Will be created if non-existing")
    parser.add_argument('--epochs', default="1000", help="Number of Epochs (Full passes through the unsupervised "
                                                         "training set)", type=int)
    parser.add_argument('--checkpoint', default="", help="Checkpoint path")
    parser.add_argument('--learningrate', default=3e-4, type=float, help="Learning rate for Adam. Defaults to "
                                                                         "Karpathy's constant ;-)")

    # Domain Adaptation Args
    parser.add_argument('--source', default="mnist", choices=['mnist', 'svhn'], help="Source Dataset. Choose mnist or "
                                                                                     "svhn")
    parser.add_argument('--target', default="svhn", choices=['mnist', 'svhn'], help="Target Dataset. Choose mnist or "
                                                                                    "svhn")

    parser.add_argument('--sourcebatch', default=100, type=int, help="Batch size of Source")
    parser.add_argument('--targetbatch', default=1000, type=int, help="Batch size of Target")

    # Associative DA Hyperparams
    parser.add_argument('--visit', default=0.1, type=float, help="Visit weight")
    parser.add_argument('--walker', default=1.0, type=float, help="Walker weight")

    return parser


if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    # Network
    if osp.exists(args.checkpoint):
        print("Resume from checkpoint file at {}".format(args.checkpoint))
        model = torch.load(args.checkpoint)
    else:
        model = Network.FrenchModel()

    # Adam optimizer, with amsgrad enabled
    optim = torch.optim.Adam(model.parameters(), lr=args.learningrate, betas=(0.5, 0.999), amsgrad=True)

    # Dataset
    datasets = Load_data.load_dataset(path="data", train=True)

    train_loader = torch.utils.data.DataLoader(datasets[args.source], batch_size=args.sourcebatch, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(datasets[args.target], batch_size=args.targetbatch, shuffle=True, num_workers=4)

    os.makedirs(args.log, exist_ok=True)
    solver.fit(model, optim, (train_loader, val_loader), n_epochs=args.epochs,
               savedir=args.log, visit_weight=args.visit, walker_weight=args.walker,
               cuda=None if args.cpu else args.gpu)
