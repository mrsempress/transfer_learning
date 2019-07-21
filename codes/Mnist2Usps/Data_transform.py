# encoding=utf-8
"""
    Created on 20:52 2019/07/20
    @author: Chenxi Huang
    This is transform the data, so the train and test data can be same size.
"""
from torchvision import datasets
import os
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import USPS as U


# Standardise it
def standardise_samples(X):
    X = X - X.mean(axis=(1, 2, 3), keepdims=True)
    X = X / X.std(axis=(1, 2, 3), keepdims=True)
    return X


# check the path is right
def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.training_file)) and \
           os.path.exists(os.path.join(self.root, self.test_file))


# load the USPS and normalize it
def load_USPS(root_dir):
    T = {
        # ToTensor(): make it in range 0 and 1
        # Normaliza(mean, std): channel =（channel - mean）/ std
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([28, 28], interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.25466308,), std=(0.3518109,))
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([28, 28], interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.26791447,), std=(0.3605367,))
        ])

    }

    USPS = {
        'train': U.USPSDataset(
            root_dir=root_dir,
            train=True,
            transform=T['train'],
        ),
        'test': U.USPSDataset(
            root_dir=root_dir,
            train=False,
            transform=T['test'],
        ),
    }
    return USPS


def load_MNIST(root_dir, resize_size=28, Gray_to_RGB=False):
    # MNIST train [60000,1,28,28] test [10000,1,28,28]
    T = {'train': [], 'test': []}

    if Gray_to_RGB:
        T['train'].append(transforms.Grayscale(num_output_channels=3))
        T['test'].append(transforms.Grayscale(num_output_channels=3))

    if resize_size == 32:
        T['train'].append(transforms.Pad(padding=2, fill=0, padding_mode='constant'))
        T['test'].append(transforms.Pad(padding=2, fill=0, padding_mode='constant'))

    T['train'].append(transforms.ToTensor())
    T['test'].append(transforms.ToTensor())

    T['train'].append(transforms.Normalize(mean=(0.1306407,), std=(0.3080536,)))
    T['test'].append(transforms.Normalize(mean=(0.13387166,), std=(0.31166542,)))

    MNIST = {
        'train': datasets.MNIST(
            root=root_dir, train=True, download=True,
            transform=transforms.Compose(T['train'])
        ),
        'test': datasets.MNIST(
            root=root_dir, train=False, download=True,
            transform=transforms.Compose(T['test'])
        )
    }
    return MNIST


def cal_mean_and_std():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    USPS = load_USPS(root_dir='../data/usps-dataset/')
    data_loader = torch.utils.data.DataLoader(
        USPS['test'],
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    data_mean = []  # Mean of the dataset
    data_std0 = []  # std of dataset
    data_std1 = []  # std with Means Delta Degrees of Freedom = 1
    for i, data in enumerate(data_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        data_mean.append(batch_mean)
        data_std0.append(batch_std0)
        data_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    data_mean = np.array(data_mean).mean(axis=0)
    data_std0 = np.array(data_std0).mean(axis=0)
    data_std1 = np.array(data_std1).mean(axis=0)

    print(data_mean, data_std0, data_std1)


if __name__ == '__main__':
    USPS = load_USPS(root_dir='../data/usps-dataset/')
    data_loader = torch.utils.data.DataLoader(
        USPS['train'],
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    print(iter(data_loader).next()[0].size())
