# encoding=utf-8
"""
    Created on
    @author: Chenxi Huang
    This is ResNet50 with Fine-tuning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import Load_data
import os


# Parameter setting
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
N_CLASS = 31  # because use Office-31
LEARNING_RATE = 1e-4
BATCH_SIZE = {'src': 256, 'tar': 256}
N_EPOCH = 100
MOMENTUM = 0.9
DECAY = 5e-4


def load_model():
    model = torchvision.models.resnet50(pretrained=True)
    n_features = model.fc.in_features
    fc = torch.nn.Linear(n_features, N_CLASS)
    model.fc = fc
    return model


def get_optimizer():
    learning_rate = LEARNING_RATE
    param_group = []
    for k, v in model.named_parameters():
        if not k.__contains__('fc'):
            param_group += [{'params': v, 'lr': learning_rate}]
        else:
            param_group += [{'params': v, 'lr': learning_rate * 10}]
    # Construct an Optimizer and optimize it with a parameter that contains a Variable object.
    # Then, specify the optimizer's parameter options, such as learning rate, weight attenuation, and so on.
    # SGD is the most basic optimization method. The ordinary training method needs to repeatedly put the
    # whole set of data into the neural network NN for training.
    # The Momentum traditional parameter W is updated by adding the original W accumulate to a
    # negative learning rate multiplied by the correction value (dx). This method compares the twists and turns.
    optimizer = optim.SGD(param_group, momentum=MOMENTUM)
    return optimizer


def finetune(model, dataloaders, optimizer):
    since = time.time()
    best_acc = 0.0
    acc_hist = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, N_EPOCH + 1):
        print('Learning rate: {:.8f}'.format(optimizer.param_groups[0]['lr']))
        print('Learning rate: {:.8f}'.format(optimizer.param_groups[-1]['lr']))
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
            if phase == 'tar' and epoch_acc > best_acc:
                best_acc = epoch_acc
        print()
    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    return model, best_acc, acc_hist


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
    model = load_model()

    # optimize it
    optimizer = get_optimizer()
    model_best, best_acc, acc_hist = finetune(model, dataloaders, optimizer)
    print('{}Best acc: {}'.format('*' * 10, best_acc))
