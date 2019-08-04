# encoding=utf-8
"""
    Created on 21:41 2019/08/01
    @author: Chenxi Huang
    Add the function of record log.
"""
import time
import os
import pandas as pd


class Log:
    def __init__(self):
        self.model_type = None
        self.source = None
        self.target = None
        self.log = {
            'time': [],
            'epoch': [],
            'source': [],
            'target': [],
            'model': [],
            'optimizer': [],
            'train_acc': [],
            'test_acc': [],
        }
        self.logs_dir = ''

    def set_dir(self, model_type, source, target):
        self.logs_dir = './results/' + model_type + '/' + source[0:1].upper() + '_to_' + target[0:1].upper()
        self.source = source
        self.target = target
        self.mode_type = model_type

    def add_log(self, epoch, optimizer_type, train_acc, test_acc):
        self.log['time'].append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.log['epoch'].append(epoch)
        self.log['source'].append(self.source)
        self.log['target'].append(self.target)
        self.log['model'].append(self.model_type)
        self.log['optimizer'].append(optimizer_type)
        self.log['train_acc'].append('%.4f' % train_acc)
        self.log['test_acc'].append('%.4f' % test_acc)

    def save_log(self):
        path = os.path.join(self.logs_dir, self.model_type + '.csv')

        log = pd.DataFrame(
            data=self.log,
            columns=['time', 'epoch', 'source', 'target', 'model', 'optimizer', 'train_acc', 'test_acc']
        )
        log.to_csv(path, mode='w', index=0)


