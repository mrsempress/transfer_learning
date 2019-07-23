# encoding=utf-8
"""
    Created on
    @author: Chenxi Huang
    This is ResNet50 with Fine-tuning.
"""
import torch
import Load_data
import os
import Fine_tune

# Parameter setting
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
N_CLASS = 31  # because use Office-31
LEARNING_RATE = 1e-4
BATCH_SIZE = {'src': 256, 'tar': 256}
N_EPOCH = 100
MOMENTUM = 0.9
DECAY = 5e-4


if __name__ == '__main__':
    torch.manual_seed(10)  # make it random
    # Load data
    root_dir = '../data/Office-31/'
    print("TCA:  data = Office-31 lambda = 1 src: " + 'webcam.mat' + " tar: " + 'dslr.mat')
    domain = {'src': str('webcam.mat'), 'tar': str('dslr.mat')}

    dataloaders = {}
    dataloaders['tar'] = Load_data.load_data(root_dir, 'tar', BATCH_SIZE['tar'], 'tar')
    dataloaders['src'], dataloaders['val'] = Load_data.load_train(root_dir, 'src', BATCH_SIZE['src'], 'src')
    print(len(dataloaders['src'].dataset), len(dataloaders['val'].dataset))

    # Load model
    model = Fine_tune.load_model('resnet')

    # optimize it
    optimizer = Fine_tune.get_optimizer('resnet')
    model_best, best_acc, acc_hist = Fine_tune.finetune(model, dataloaders, optimizer)
    print('{}Best acc: {}'.format('*' * 10, best_acc))
