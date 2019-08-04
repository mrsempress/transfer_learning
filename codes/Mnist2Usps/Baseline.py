# encoding=utf-8
"""
    Created on 13:18 2019/07/20
    @author: Chenxi Huang
    This is Mnist to Usps using Self-ensemble D's network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import Data_transform
import Network
import os
import Log


def train(model, dataloaders, optimizer, log, N_EPOCH=25):
    since = time.time()
    best_acc = 0.0
    acc_hist = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, N_EPOCH + 1):
        for phase in ['src', 'val', 'tar']:
            if phase == 'src':
                model.train()  # Model.train() : Enable BatchNormalization and Dropout, compute the value
            else:
                model.eval()  # Model.eval() : Do not enable BatchNormalization and Dropout, fixed the value
            total_loss, correct = 0, 0
            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()  # Initialize the gradient to zero
                with torch.set_grad_enabled(phase == 'src'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'src':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct.double() / len(dataloaders[phase].dataset)
            acc_hist.append([epoch_loss, epoch_acc])
            print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, N_EPOCH, phase, epoch_loss,
                                                                                  epoch_acc))

            # add log
            log.add_log(epoch, 'Adam', '*', epoch_acc)

            if phase == 'tar' and epoch_acc > best_acc:
                best_acc = epoch_acc
        print()
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    return model, best_acc, acc_hist


def work(source, target, gpu, n_classes=10, batch_size=256, num_workers=4, lr=0.001, N_EPOCH=25):
    # set log information
    log = Log.Log()
    log.set_dir('Baseline_Digit', source, target)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if source == 'MNIST':
        source_data = Data_transform.load_MNIST(root_dir='data/Digits/MNIST', Gray_to_RGB=False)
    elif source == 'USPS':
        source_data = Data_transform.load_USPS(root_dir='data/Digits/USPS')
    else:
        source_data = Data_transform.load_SVHN(root_dir='data/Digits/SVHN')

    if target == 'MNIST':
        target_data = Data_transform.load_MNIST(root_dir='data/Digits/MNIST', Gray_to_RGB=False)
    elif target == 'USPS':
        target_data = Data_transform.load_USPS(root_dir='data/Digits/USPS')
    else:
        target_data = Data_transform.load_SVHN(root_dir='data/Digits/SVHN')

    source_data_loader = {
        'train': torch.utils.data.DataLoader(source_data['train'], batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers),
        'test': torch.utils.data.DataLoader(source_data['test'], batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
    }
    target_data_loader = {
        'train': torch.utils.data.DataLoader(target_data['train'], batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers),
        'test': torch.utils.data.DataLoader(target_data['test'], batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers)
    }

    model = Network.BaselineM2U(n_classes=n_classes)

    # Optimized all the parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = train(
        model=model,
        dataloaders={
            'src': source_data_loader['train'],
            'val': source_data_loader['test'],
            'tar': target_data_loader['train']
        },
        optimizer=optimizer,
        log=log,
        N_EPOCH=N_EPOCH
    )
    # save log
    log.save_log()


if __name__ == '__main__':
    work('MNIST', 'USPS', '3')
