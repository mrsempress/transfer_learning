from __future__ import print_function, division
import torch.nn as nn
import time
from data_helper import *
import sys
from solvers.Solver import Solver
from network import MADA
import numpy as np


class MADASolver(Solver):

    def __init__(self, dataset_type, source_domain, target_domain, cuda='cuda:0',
                 pretrained=False,
                 batch_size=32,
                 num_epochs=9999, max_iter_num=9999999, test_interval=500, test_mode=False, num_workers=2,
                 clean_log=False, lr=0.001, gamma=10, optimizer_type='SGD'):
        super(MADASolver, self).__init__(
            dataset_type=dataset_type,
            source_domain=source_domain,
            target_domain=target_domain,
            cuda=cuda,
            pretrained=pretrained,
            batch_size=batch_size,
            num_epochs=num_epochs,
            max_iter_num=max_iter_num,
            test_interval=test_interval,
            test_mode=test_mode,
            num_workers=num_workers,
            clean_log=clean_log,
            lr=lr,
            gamma=gamma,
            optimizer_type=optimizer_type
        )
        self.model_name = 'MADA'
        self.iter_num = 0
        self.class_weight = None

    def get_alpha(self, delta=10.0):
        if self.num_epochs != 999999:
            p = self.epoch / self.num_epochs
        else:
            p = self.iter_num / self.max_iter_num

        return np.float(2.0 / (1.0 + np.exp(-delta * p)) - 1.0)

    def set_model(self):
        if self.dataset_type == 'Digits':
            if self.task in ['MtoU', 'UtoM']:
                self.model = MADA(n_classes=self.n_classes, base_model='DigitsMU')
            if self.task in ['StoM']:
                self.model = MADA(n_classes=self.n_classes, base_model='DigitsStoM')

        if self.dataset_type in ['Office31', 'OfficeHome']:
            self.model = MADA(n_classes=self.n_classes, base_model='ResNet50')

        if self.pretrained:
            self.load_model(path=self.models_checkpoints_dir + '/' + self.model_name + '_best_train.pt')

        self.model = self.model.to(self.device)

    def test(self, data_loader):
        self.model.eval()

        total_loss = 0
        corrects = 0
        data_num = len(data_loader.dataset)
        processed_num = 0

        for inputs, labels in data_loader:
            sys.stdout.write('\r{}/{}'.format(processed_num, data_num))
            sys.stdout.flush()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # print('inputs ',inputs.size())
            # print('labels ',labels.size())

            class_outputs = self.model(inputs, test_mode=True)

            # print('class outputs ',class_outputs.size())

            _, preds = torch.max(class_outputs, 1)
            # print('preds ',preds.size())

            # loss = nn.CrossEntropyLoss()(class_outputs, labels)

            # total_loss += loss.item() * labels.size()[0]
            corrects += (preds == labels.data).sum().item()
            processed_num += labels.size()[0]

            # self.model.train(False)
            # self.class_weight = torch.mean(nn.Softmax(dim=1)(class_outputs), 0)
            # self.class_weight = (self.class_weight / torch.mean(self.class_weight))
            # self.class_weight = self.class_weight.view(-1)
            # self.class_weight = self.class_weight.detach()

        acc = corrects / processed_num
        average_loss = total_loss / processed_num
        print('\nData size = {} , corrects = {}'.format(processed_num, corrects))

        return average_loss, acc

    def train_one_epoch(self):
        since = time.time()
        self.model.train()

        total_loss = 0
        source_corrects = 0

        total_target_num = len(self.data_loader['target']['train'].dataset)
        processed_target_num = 0
        total_source_num = 0

        # class_criterion = nn.CrossEntropyLoss(weight=self.class_weight.view(-1))
        class_criterion = nn.CrossEntropyLoss()

        alpha = 0
        for target_inputs, target_labels in self.data_loader['target']['train']:
            sys.stdout.write('\r{}/{}'.format(processed_target_num, total_target_num))
            sys.stdout.flush()

            self.update_optimizer()

            self.optimizer.zero_grad()

            alpha = self.get_alpha()

            # TODO 1 : Target Train

            target_inputs = target_inputs.to(self.device)

            target_domain_outputs, target_class_outputs = self.model(target_inputs, alpha=alpha)

            target_domain_labels = torch.ones((target_labels.size()[0] * self.n_classes, 1), device=self.device)

            target_domain_loss = nn.BCELoss()(target_domain_outputs.view(-1), target_domain_labels.view(-1))

            # TODO 2 : Source Train

            source_iter = iter(self.data_loader['source']['train'])
            source_inputs, source_labels = next(source_iter)

            source_inputs = source_inputs.to(self.device)
            source_domain_outputs, source_class_outputs = self.model(source_inputs, alpha=alpha)

            source_labels = source_labels.to(self.device)
            source_class_loss = class_criterion(source_class_outputs, source_labels)

            source_domain_labels = torch.zeros((source_labels.size()[0] * self.n_classes, 1), device=self.device)

            source_domain_loss = nn.BCELoss()(source_domain_outputs.view(-1), source_domain_labels.view(-1))

            # TODO 3 : LOSS

            # loss_weight = torch.Tensor(self.n_classes,device=self.device)
            loss = self.n_classes * (target_domain_loss + source_domain_loss) + source_class_loss

            loss.backward()

            self.optimizer.step()

            # TODO 5 : other parameters
            total_loss += loss.item() * source_labels.size()[0]
            _, source_class_preds = torch.max(source_class_outputs, 1)
            source_corrects += (source_class_preds == source_labels.data).sum().item()
            total_source_num += source_labels.size()[0]
            processed_target_num += target_labels.size()[0]
            self.iter_num += 1

        acc = source_corrects / total_source_num
        average_loss = total_loss / total_source_num

        print()
        print('\nData size = {} , corrects = {}'.format(total_source_num, source_corrects))
        print('Using {:4f}'.format(time.time() - since))
        print('Alpha = ', alpha)
        return average_loss, acc