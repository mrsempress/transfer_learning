# encoding=utf-8
"""
    Created on
    @author: Chenxi Huang
    This is ResNet50 with Fine-tuning.
"""
import torch
import Load_data
import os
from Fine_tune import Fine_tune


def work(source, target, _model='resnet', gpu='3', seed=10, batch_size=256):
    # Parameter setting
    DEVICE = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = {'src': batch_size, 'tar': batch_size}

    torch.manual_seed(seed)  # make it random
    # Load data
    root_dir = 'data/Office-31/'
    domain = {'src': str(source+'.mat'), 'tar': str(target+'.mat')}

    dataloaders = {}
    dataloaders['tar'] = Load_data.load_data(root_dir, 'tar', BATCH_SIZE['tar'], 'tar')
    dataloaders['src'], dataloaders['val'] = Load_data.load_train(root_dir, 'src', BATCH_SIZE['src'], 'src')
    print(len(dataloaders['src'].dataset), len(dataloaders['val'].dataset))

    # Load model
    Fine_tune.call(gpu, batchsize=batch_size)

    model = Fine_tune.load_model(_model).to(DEVICE)

    # optimize it
    optimizer = Fine_tune.get_optimizer(model, _model)
    model_best, best_acc, acc_hist = Fine_tune.finetune(model, dataloaders, optimizer)
    print('{}Best acc: {}'.format('*' * 10, best_acc))


if __name__ == '__main__':
    work('webcam', 'dslr')
