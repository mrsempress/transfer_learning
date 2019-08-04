# encoding=utf-8
"""
    Created on 15:30 2019/07/29
    @author: Chenxi Huang
    It solve the MADA, including train, test.
"""
import torch
import torch.nn as nn
import numpy as np


def test(model, data_loader):
    model.eval()
    total_loss = 0
    corrects = 0
    processed_num = 0

    for inputs, labels in data_loader:
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        class_outputs = model(inputs, test_mode=True)

        _, preds = torch.max(class_outputs, 1)

        # loss = nn.CrossEntropyLoss()(class_outputs, labels)
        # total_loss += loss.item() * labels.size()[0]
        corrects += (preds == labels.data).sum().item()
        processed_num += labels.size()[0]

    acc = corrects / processed_num
    average_loss = total_loss / processed_num
    print('Data size = {} , corrects = {}'.format(processed_num, corrects))
    return average_loss, acc


def train_one_epoch(model, data_loader, optimizer, n_classes,
                    optimizer_type, num_epochs, epoch, iter_num, max_iter_num, lr, gamma, loss_weight):
    model.train()

    total_loss = 0
    source_corrects = 0

    processed_target_num = 0
    total_source_num = 0

    class_criterion = nn.CrossEntropyLoss()

    alpha = 0
    for target_inputs, target_labels in data_loader['target']['train']:
        cur_lr = update_optimizer(optimizer, optimizer_type, num_epochs, epoch, iter_num, max_iter_num, lr, gamma)

        optimizer.zero_grad()

        alpha = get_alpha(num_epochs, epoch, iter_num, max_iter_num)

        # Target Train
        # target_inputs = target_inputs.to(device)
        target_domain_outputs, target_class_outputs = model(target_inputs, alpha=alpha)
        # target_domain_labels = torch.ones((target_labels.size()[0] * n_classes, 1), device=device)
        target_domain_labels = torch.ones((target_labels.size()[0] * n_classes, 1))
        target_domain_loss = nn.BCELoss()(target_domain_outputs.view(-1), target_domain_labels.view(-1))

        # Source Train
        source_iter = iter(data_loader['source']['train'])
        source_inputs, source_labels = next(source_iter)
        # source_inputs = source_inputs.to(device)
        source_domain_outputs, source_class_outputs = model(source_inputs, alpha=alpha)
        # source_labels = source_labels.to(device)
        source_class_loss = class_criterion(source_class_outputs, source_labels)
        # source_domain_labels = torch.zeros((source_labels.size()[0] * n_classes, 1), device=device)
        source_domain_labels = torch.zeros((source_labels.size()[0] * n_classes, 1))
        source_domain_loss = nn.BCELoss()(source_domain_outputs.view(-1), source_domain_labels.view(-1))

        # LOSS
        # loss = target_domain_loss + source_domain_loss + source_class_loss
        loss = loss_weight * 0.5 * n_classes * (target_domain_loss + source_domain_loss) + source_class_loss
        loss.backward()
        optimizer.step()

        # Other parameters
        total_loss += loss.item() * source_labels.size()[0]
        _, source_class_preds = torch.max(source_class_outputs, 1)
        source_corrects += (source_class_preds == source_labels.data).sum().item()
        total_source_num += source_labels.size()[0]
        processed_target_num += target_labels.size()[0]
        iter_num += 1

    acc = source_corrects / total_source_num
    average_loss = total_loss / total_source_num

    print('Data size = {} , corrects = {}'.format(total_source_num, source_corrects))
    print('Alpha = ', alpha)
    print()
    return average_loss, acc, iter_num, cur_lr


def train(model, data_loader, optimizer, optimizer_type, test_interval,
          max_iter_num, num_epochs, n_classes, lr, gamma, loss_weight, log):
    iter_num = 0  # Iteration is the number of batches that an epoch needs.
    log_iter = 0

    best_val_loss, best_val_acc = test(model, data_loader['source']['test'])

    print('Initial Train Loss: {:.4f} Acc: {:.4f}'.format(best_val_loss, best_val_acc))

    best_test_loss, best_test_acc = test(model, data_loader['target']['test'])
    print('Initial Test Loss: {:.4f} Acc: {:.4f}'.format(best_val_loss, best_val_acc))

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('iteration : {}'.format(iter_num))

        # train
        train_loss, train_acc, iter_num, cur_lr = train_one_epoch(model, data_loader, optimizer, n_classes,
                                                optimizer_type, num_epochs, epoch, iter_num, max_iter_num,
                                                lr, gamma, loss_weight)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
        if train_acc >= best_val_acc:
            best_val_acc = train_acc

        val_acc = val_loss = 0

        # Test
        # if iter_num - log_iter >= test_interval:
        #     log_iter = iter_num
        test_loss, test_acc = test(model, data_loader['target']['test'])
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
        log.add_log(epoch, optimizer_type, train_acc, test_acc)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
        print('Current Best Test Acc : {:4f}    Current Best Test Loss : {:4f}  Cur lr : {:4f}'.format(best_val_acc,
                                                                                                      best_test_loss,
                                                                                                      cur_lr))
        if iter_num >= max_iter_num:
            break

    print('Best Val Acc : {:4f}, Test Acc : {:4f}'.format(best_val_acc, best_test_acc))


def update_optimizer(optimizer, optimizer_type, num_epochs, epoch, iter_num, max_iter_num, lr, gamma,
                     power=0.75, weight_decay=0.0005):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    """
    if optimizer_type == 'SGD':
        if num_epochs != 999999:
            p = epoch / num_epochs
        else:
            p = iter_num / max_iter_num

        lr = lr * (1.0 + gamma * p) ** (-power)
    else:
        lr = lr

    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
    return cur_lr


def get_alpha(num_epochs, epoch, iter_num, max_iter_num, delta=10.0):
    if num_epochs != 999999:
        p = epoch / num_epochs
    else:
        p = iter_num / max_iter_num

    return np.float(2.0 / (1.0 + np.exp(-delta * p)) - 1.0)
