# encoding=utf-8
"""
    Created on 15:31 2019/07/29
    @author: Chenxi Huang
    It implements "Multi-Adversarial Domain Adaptation".
"""
import torch
import Network
import os
import Load_data
from MADA import solver
from torch.utils.data import DataLoader


def work(sources, targets, gpu, batch_size=32, num_epochs=256, lr=0.001, gamma=10, optimizer_type='SGD',
         test_interval=500, max_iter_num=256, weight_decay=0.0005, momentum=0.9, num_workers=2, loss_weight=1.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # load dataset
    n_classes = 31

    # Load data
    root_dir = 'data/Office-31/'
    # domain = {'src': str(source + '.mat'), 'tar': str(target + '.mat')}
    # BATCH_SIZE = {'src': batch_size, 'tar': batch_size}
    source_data = Load_data.load_Office(root_dir + 'src/', sources)
    target_data = Load_data.load_Office(root_dir + 'tar/', targets)
    
    # set dataloader
    data_loader = {
        'source': {
            'train': None,
            'test': None
        },
        'target': {
            'train': None,
            'test': None
        }
    }
    data_loader['source']['train'] = DataLoader(
        dataset=source_data['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    data_loader['source']['test'] = DataLoader(
        source_data['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    data_loader['target']['train'] = DataLoader(
        target_data['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    data_loader['target']['test'] = DataLoader(
        target_data['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # set model
    model = Network.MADA(n_classes, True)

    # Set optimizer
    if optimizer_type == 'Adam':
        optimizer_type = 'Adam'
        optimizer = torch.optim.Adam(
            params=model.get_parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer_type = 'SGD'
        optimizer = torch.optim.SGD(
            params=model.get_parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )

    # train
    solver.train(model, data_loader, optimizer, optimizer_type, test_interval,
          max_iter_num, num_epochs, n_classes, lr, gamma, loss_weight)


if __name__ == '__main__':
    work('amazon', 'webcam', '3')
