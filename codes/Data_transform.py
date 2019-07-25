# encoding=utf-8
"""
    Created on 20:52 2019/07/20
    @author: Chenxi Huang
    This is transform the data, so the train and test data can be same size or satisfy the experiment requirements
"""
from torchvision import datasets
import os
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import Dataset as U
from skimage.transform import downscale_local_mean, resize
from batchup.datasets import mnist, fashion_mnist, cifar10, svhn, stl, usps
import Dataset
from scipy.io import loadmat
import utils
import pickle as pkl
import gzip
import importlib


# Standardise samples
def standardise_samples(X):
    X = X - X.mean(axis=(1, 2, 3), keepdims=True)
    X = X / X.std(axis=(1, 2, 3), keepdims=True)
    return X


# Standardise dataset
def standardise_dataset(ds):
    ds.train_X_orig = ds.train_X[...].copy()
    ds.val_X_orig = ds.val_X[...].copy()
    ds.test_X_orig = ds.test_X[...].copy()

    ds.train_X = standardise_samples(ds.train_X[...])
    ds.val_X = standardise_samples(ds.val_X[...])
    ds.test_X = standardise_samples(ds.test_X[...])


# check the path is right
def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.training_file)) and \
           os.path.exists(os.path.join(self.root, self.test_file))


# rgb to grey
def rgb2grey_tensor(X):
    return (X[:, 0:1, :, :] * 0.2125) + (X[:, 1:2, :, :] * 0.7154) + (X[:, 2:3, :, :] * 0.0721)


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


def load_usps(invert=False, zero_centre=False, val=False, scale28=False):
    # Load USPS
    print('Loading USPS...')

    if val:
        d_usps = usps.USPS()
    else:
        d_usps = usps.USPS(n_val=None)

    d_usps.train_X = d_usps.train_X[:]
    d_usps.val_X = d_usps.val_X[:]
    d_usps.test_X = d_usps.test_X[:]
    d_usps.train_y = d_usps.train_y[:]
    d_usps.val_y = d_usps.val_y[:]
    d_usps.test_y = d_usps.test_y[:]

    if scale28:
        def _resize_tensor(X):
            X_prime = np.zeros((X.shape[0], 1, 28, 28), dtype=np.float32)
            for i in range(X.shape[0]):
                X_prime[i, 0, :, :] = resize(X[i, 0, :, :], (28, 28), mode='constant')
            return X_prime

        # Scale 16x16 to 28x28
        d_usps.train_X = _resize_tensor(d_usps.train_X)
        d_usps.val_X = _resize_tensor(d_usps.val_X)
        d_usps.test_X = _resize_tensor(d_usps.test_X)

    if invert:
        # Invert
        d_usps.train_X = 1.0 - d_usps.train_X
        d_usps.val_X = 1.0 - d_usps.val_X
        d_usps.test_X = 1.0 - d_usps.test_X

    if zero_centre:
        d_usps.train_X = d_usps.train_X * 2.0 - 1.0
        d_usps.test_X = d_usps.test_X * 2.0 - 1.0

    print('USPS: train: \nX.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, \ntest: X.shape={}, y.shape={}'.format(
        d_usps.train_X.shape, d_usps.train_y.shape,
        d_usps.val_X.shape, d_usps.val_y.shape,
        d_usps.test_X.shape, d_usps.test_y.shape))

    print('USPS: train: X.min={}, X.max={}'.format(
        d_usps.train_X.min(), d_usps.train_X.max()))

    d_usps.n_classes = 10

    return d_usps


def load_usps_32(path='data/usps_28x28.pkl', all_use=False):
    f = gzip.open(path, 'rb')
    data_set = pkl.load(f, encoding='bytes')
    f.close()
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    if all_use == 'yes':
        img_train = img_train[inds][:6562]
        label_train = label_train[inds][:6562]
    else:
        img_train = img_train[inds][:1800]
        label_train = label_train[inds][:1800]
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))
    return img_train, label_train, img_test, label_test


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


def load_mnist_32(path='../data/mnist_data.mat', scale=True, usps=False, all_use=False):
    mnist_data = loadmat(path)
    if scale:
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test =  mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    if usps and all_use != 'yes':
        mnist_train = mnist_train[:2000]
        train_label = train_label[:2000]

    return mnist_train, train_label, mnist_test, test_label


def load_mnist(invert=False, zero_centre=False, intensity_scale=1.0, val=False, pad32=False, downscale_x=1,
               rgb=False):
    # Load MNIST
    print('Loading MNIST...')

    if val:
        d_mnist = mnist.MNIST(n_val=10000)
    else:
        d_mnist = mnist.MNIST(n_val=0)

    d_mnist.train_X = d_mnist.train_X[:]
    d_mnist.val_X = d_mnist.val_X[:]
    d_mnist.test_X = d_mnist.test_X[:]
    d_mnist.train_y = d_mnist.train_y[:]
    d_mnist.val_y = d_mnist.val_y[:]
    d_mnist.test_y = d_mnist.test_y[:]

    if downscale_x != 1:
        d_mnist.train_X = downscale_local_mean(d_mnist.train_X, (1, 1, 1, downscale_x))
        d_mnist.val_X = downscale_local_mean(d_mnist.val_X, (1, 1, 1, downscale_x))
        d_mnist.test_X = downscale_local_mean(d_mnist.test_X, (1, 1, 1, downscale_x))

    if pad32:
        py = (32 - d_mnist.train_X.shape[2]) // 2
        px = (32 - d_mnist.train_X.shape[3]) // 2
        # Pad 28x28 to 32x32
        d_mnist.train_X = np.pad(d_mnist.train_X, [(0, 0), (0, 0), (py, py), (px, px)], mode='constant')
        d_mnist.val_X = np.pad(d_mnist.val_X, [(0, 0), (0, 0), (py, py), (px, px)], mode='constant')
        d_mnist.test_X = np.pad(d_mnist.test_X, [(0, 0), (0, 0), (py, py), (px, px)], mode='constant')

    if invert:
        # Invert
        d_mnist.train_X = 1.0 - d_mnist.train_X
        d_mnist.val_X = 1.0 - d_mnist.val_X
        d_mnist.test_X = 1.0 - d_mnist.test_X

    if intensity_scale != 1.0:
        d_mnist.train_X = (d_mnist.train_X - 0.5) * intensity_scale + 0.5
        d_mnist.val_X = (d_mnist.val_X - 0.5) * intensity_scale + 0.5
        d_mnist.test_X = (d_mnist.test_X - 0.5) * intensity_scale + 0.5

    if zero_centre:
        d_mnist.train_X = d_mnist.train_X * 2.0 - 1.0
        d_mnist.test_X = d_mnist.test_X * 2.0 - 1.0

    if rgb:
        d_mnist.train_X = np.concatenate([d_mnist.train_X] * 3, axis=1)
        d_mnist.val_X = np.concatenate([d_mnist.val_X] * 3, axis=1)
        d_mnist.test_X = np.concatenate([d_mnist.test_X] * 3, axis=1)

    print('MNIST: \ntrain: X.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, \ntest: X.shape={}, y.shape={}'.format(
        d_mnist.train_X.shape, d_mnist.train_y.shape,
        d_mnist.val_X.shape, d_mnist.val_y.shape,
        d_mnist.test_X.shape, d_mnist.test_y.shape))

    print('MNIST: train: X.min={}, X.max={}'.format(
        d_mnist.train_X.min(), d_mnist.train_X.max()))

    d_mnist.n_classes = 10

    return d_mnist


def load_fashion_mnist(invert=False, zero_centre=False, intensity_scale=1.0, val=False, pad32=False, downscale_x=1):
    # It is a positive image of 70,000 different products in 10 categories. More complicated than handwritten numbers.
    # Load MNIST
    print('Loading Fashion MNIST...')

    if val:
        d_fmnist = fashion_mnist.FashionMNIST(n_val=10000)
    else:
        d_fmnist = fashion_mnist.FashionMNIST(n_val=0)

    d_fmnist.train_X = d_fmnist.train_X[:]
    d_fmnist.val_X = d_fmnist.val_X[:]
    d_fmnist.test_X = d_fmnist.test_X[:]
    d_fmnist.train_y = d_fmnist.train_y[:]
    d_fmnist.val_y = d_fmnist.val_y[:]
    d_fmnist.test_y = d_fmnist.test_y[:]

    if downscale_x != 1:
        d_fmnist.train_X = downscale_local_mean(d_fmnist.train_X, (1, 1, 1, downscale_x))
        d_fmnist.val_X = downscale_local_mean(d_fmnist.val_X, (1, 1, 1, downscale_x))
        d_fmnist.test_X = downscale_local_mean(d_fmnist.test_X, (1, 1, 1, downscale_x))

    if pad32:
        py = (32 - d_fmnist.train_X.shape[2]) // 2
        px = (32 - d_fmnist.train_X.shape[3]) // 2
        # Pad 28x28 to 32x32
        d_fmnist.train_X = np.pad(d_fmnist.train_X, [(0, 0), (0, 0), (py, py), (px, px)], mode='constant')
        d_fmnist.val_X = np.pad(d_fmnist.val_X, [(0, 0), (0, 0), (py, py), (px, px)], mode='constant')
        d_fmnist.test_X = np.pad(d_fmnist.test_X, [(0, 0), (0, 0), (py, py), (px, px)], mode='constant')

    if invert:
        # Invert
        d_fmnist.train_X = 1.0 - d_fmnist.train_X
        d_fmnist.val_X = 1.0 - d_fmnist.val_X
        d_fmnist.test_X = 1.0 - d_fmnist.test_X

    if intensity_scale != 1.0:
        d_fmnist.train_X = (d_fmnist.train_X - 0.5) * intensity_scale + 0.5
        d_fmnist.val_X = (d_fmnist.val_X - 0.5) * intensity_scale + 0.5
        d_fmnist.test_X = (d_fmnist.test_X - 0.5) * intensity_scale + 0.5

    if zero_centre:
        d_fmnist.train_X = d_fmnist.train_X * 2.0 - 1.0
        d_fmnist.test_X = d_fmnist.test_X * 2.0 - 1.0

    print('Fashion MNIST: \ntrain: X.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, '
          'test: X.shape={}, y.shape={}'.format(
        d_fmnist.train_X.shape, d_fmnist.train_y.shape,
        d_fmnist.val_X.shape, d_fmnist.val_y.shape,
        d_fmnist.test_X.shape, d_fmnist.test_y.shape))

    print('Fashion MNIST: train: X.min={}, X.max={}'.format(
        d_fmnist.train_X.min(), d_fmnist.train_X.max()))

    d_fmnist.n_classes = 10

    return d_fmnist


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


# Dataset loading functions
def load_svhn(zero_centre=False, greyscale=False, val=False, extra=False):
    # Load SVHN
    print('Loading SVHN...')
    if val:
        d_svhn = svhn.SVHN(n_val=10000)
    else:
        d_svhn = svhn.SVHN(n_val=0)

    if extra:
        d_extra = svhn.SVHNExtra()
    else:
        d_extra = None

    d_svhn.train_X = d_svhn.train_X[:]
    d_svhn.val_X = d_svhn.val_X[:]
    d_svhn.test_X = d_svhn.test_X[:]
    d_svhn.train_y = d_svhn.train_y[:]
    d_svhn.val_y = d_svhn.val_y[:]
    d_svhn.test_y = d_svhn.test_y[:]

    if extra:
        d_svhn.train_X = np.append(d_svhn.train_X, d_extra.X[:], axis=0)
        d_svhn.train_y = np.append(d_svhn.train_y, d_extra.y[:], axis=0)

    if greyscale:
        d_svhn.train_X = rgb2grey_tensor(d_svhn.train_X)
        d_svhn.val_X = rgb2grey_tensor(d_svhn.val_X)
        d_svhn.test_X = rgb2grey_tensor(d_svhn.test_X)

    if zero_centre:
        d_svhn.train_X = d_svhn.train_X * 2.0 - 1.0
        d_svhn.val_X = d_svhn.val_X * 2.0 - 1.0
        d_svhn.test_X = d_svhn.test_X * 2.0 - 1.0

    print('SVHN: train: \nX.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, test: X.shape={}, y.shape={}'.format(
        d_svhn.train_X.shape, d_svhn.train_y.shape, d_svhn.val_X.shape, d_svhn.val_y.shape, d_svhn.test_X.shape,
        d_svhn.test_y.shape))

    print('SVHN: train: X.min={}, X.max={}'.format(
        d_svhn.train_X.min(), d_svhn.train_X.max()))

    d_svhn.n_classes = 10

    return d_svhn


def load_svhn_32(train='../data/train_32x32.mat', test='../data/test_32x32.mat'):
    svhn_train = loadmat(train)
    svhn_test = loadmat(test)
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = utils.dense_to_one_hot(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = utils.dense_to_one_hot(svhn_test['y'])

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test


def load_syn_digits(zero_centre=False, greyscale=False, val=False):
    # Load syn digits
    print('Loading Syn-digits...')
    if val:
        d_synd = Dataset.SynDigits(n_val=10000)
    else:
        d_synd = Dataset.SynDigits(n_val=0)

    d_synd.train_X = d_synd.train_X[:]
    d_synd.val_X = d_synd.val_X[:]
    d_synd.test_X = d_synd.test_X[:]
    d_synd.train_y = d_synd.train_y[:]
    d_synd.val_y = d_synd.val_y[:]
    d_synd.test_y = d_synd.test_y[:]

    if greyscale:
        d_synd.train_X = rgb2grey_tensor(d_synd.train_X)
        d_synd.val_X = rgb2grey_tensor(d_synd.val_X)
        d_synd.test_X = rgb2grey_tensor(d_synd.test_X)

    if zero_centre:
        d_synd.train_X = d_synd.train_X * 2.0 - 1.0
        d_synd.val_X = d_synd.val_X * 2.0 - 1.0
        d_synd.test_X = d_synd.test_X * 2.0 - 1.0

    print('SynDigits: \ntrain: X.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, \ntest: X.shape={}, y.shape={}'.format(
        d_synd.train_X.shape, d_synd.train_y.shape, d_synd.val_X.shape, d_synd.val_y.shape, d_synd.test_X.shape,
        d_synd.test_y.shape))

    print('SynDigits: train: X.min={}, X.max={}'.format(
        d_synd.train_X.min(), d_synd.train_X.max()))

    d_synd.n_classes = 10

    return d_synd


def load_syn_signs(zero_centre=False, greyscale=False, val=False):
    # Load syn digits
    print('Loading Syn-signs...')
    if val:
        d_syns = Dataset.SynSigns(n_val=10000, n_test=10000)
    else:
        d_syns = Dataset.SynSigns(n_val=0, n_test=10000)

    d_syns.train_X = d_syns.train_X[:]
    d_syns.val_X = d_syns.val_X[:]
    d_syns.test_X = d_syns.test_X[:]
    d_syns.train_y = d_syns.train_y[:]
    d_syns.val_y = d_syns.val_y[:]
    d_syns.test_y = d_syns.test_y[:]

    if greyscale:
        d_syns.train_X = rgb2grey_tensor(d_syns.train_X)
        d_syns.val_X = rgb2grey_tensor(d_syns.val_X)
        d_syns.test_X = rgb2grey_tensor(d_syns.test_X)

    if zero_centre:
        d_syns.train_X = d_syns.train_X * 2.0 - 1.0
        d_syns.val_X = d_syns.val_X * 2.0 - 1.0
        d_syns.test_X = d_syns.test_X * 2.0 - 1.0

    print('SynSigns: \ntrain: X.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, '
          '\ntest: X.shape={}, y.shape={}'.format(
        d_syns.train_X.shape, d_syns.train_y.shape, d_syns.val_X.shape, d_syns.val_y.shape, d_syns.test_X.shape,
        d_syns.test_y.shape))

    print('SynSigns: train: X.min={}, X.max={}'.format(
        d_syns.train_X.min(), d_syns.train_X.max()))

    d_syns.n_classes = 43

    return d_syns


def load_syntraffic(path='../data/data_synthetic'):
    data_source = pkl.load(open(path), encoding='bytes')
    source_train = np.random.permutation(len(data_source['image']))
    data_s_im = data_source['image'][source_train[:len(data_source['image'])], :, :, :]
    data_s_im_test = data_source['image'][source_train[len(data_source['image']) - 2000:], :, :, :]
    data_s_label = data_source['label'][source_train[:len(data_source['image'])]]
    data_s_label_test = data_source['label'][source_train[len(data_source['image']) - 2000:]]
    data_s_im = data_s_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_s_im_test = data_s_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    return data_s_im, data_s_label, data_s_im_test, data_s_label_test


def load_cifar10(range_01=False, val=False):
    # Load CIFAR-10 for adaptation with STL
    print('Loading CIFAR-10...')
    if val:
        d_cifar = cifar10.CIFAR10(n_val=5000)
    else:
        d_cifar = cifar10.CIFAR10(n_val=0)

    d_cifar.train_X = d_cifar.train_X[:]
    d_cifar.val_X = d_cifar.val_X[:]
    d_cifar.test_X = d_cifar.test_X[:]
    d_cifar.train_y = d_cifar.train_y[:]
    d_cifar.val_y = d_cifar.val_y[:]
    d_cifar.test_y = d_cifar.test_y[:]

    # Remap class indices so that the frog class (6) has an index of -1 as it does not appear int the STL dataset
    cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
    d_cifar.train_y = cls_mapping[d_cifar.train_y]
    d_cifar.val_y = cls_mapping[d_cifar.val_y]
    d_cifar.test_y = cls_mapping[d_cifar.test_y]

    # Remove all samples from skipped classes
    train_mask = d_cifar.train_y != -1
    val_mask = d_cifar.val_y != -1
    test_mask = d_cifar.test_y != -1

    d_cifar.train_X = d_cifar.train_X[train_mask]
    d_cifar.train_y = d_cifar.train_y[train_mask]
    d_cifar.val_X = d_cifar.val_X[val_mask]
    d_cifar.val_y = d_cifar.val_y[val_mask]
    d_cifar.test_X = d_cifar.test_X[test_mask]
    d_cifar.test_y = d_cifar.test_y[test_mask]

    if range_01:
        d_cifar.train_X = d_cifar.train_X * 2.0 - 1.0
        d_cifar.val_X = d_cifar.val_X * 2.0 - 1.0
        d_cifar.test_X = d_cifar.test_X * 2.0 - 1.0

    print('CIFAR-10: \ntrain: X.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, \ntest: X.shape={}, y.shape={}'.format(
        d_cifar.train_X.shape, d_cifar.train_y.shape, d_cifar.val_X.shape, d_cifar.val_y.shape, d_cifar.test_X.shape,
        d_cifar.test_y.shape))

    print('CIFAR-10: train: X.min={}, X.max={}'.format(
        d_cifar.train_X.min(), d_cifar.train_X.max()))

    d_cifar.n_classes = 9

    return d_cifar


def load_stl(zero_centre=False, val=False):
    # Load STL for adaptation with CIFAR-10
    print('Loading STL...')
    if val:
        d_stl = stl.STL()
    else:
        d_stl = stl.STL(n_val_folds=0)

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]
    d_stl.train_y = d_stl.train_y[:]
    d_stl.val_y = d_stl.val_y[:]
    d_stl.test_y = d_stl.test_y[:]

    # Remap class indices to match CIFAR-10:
    cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
    d_stl.train_y = cls_mapping[d_stl.train_y]
    d_stl.val_y = cls_mapping[d_stl.val_y]
    d_stl.test_y = cls_mapping[d_stl.test_y]

    d_stl.train_X = d_stl.train_X[:]
    d_stl.val_X = d_stl.val_X[:]
    d_stl.test_X = d_stl.test_X[:]

    # Remove all samples from class -1 (monkey) as it does not appear int the CIFAR-10 dataset
    train_mask = d_stl.train_y != -1
    val_mask = d_stl.val_y != -1
    test_mask = d_stl.test_y != -1

    d_stl.train_X = d_stl.train_X[train_mask]
    d_stl.train_y = d_stl.train_y[train_mask]
    d_stl.val_X = d_stl.val_X[val_mask]
    d_stl.val_y = d_stl.val_y[val_mask]
    d_stl.test_X = d_stl.test_X[test_mask]
    d_stl.test_y = d_stl.test_y[test_mask]

    # Downsample images from 96x96 to 32x32
    d_stl.train_X = downscale_local_mean(d_stl.train_X, (1, 1, 3, 3))
    d_stl.val_X = downscale_local_mean(d_stl.val_X, (1, 1, 3, 3))
    d_stl.test_X = downscale_local_mean(d_stl.test_X, (1, 1, 3, 3))

    if zero_centre:
        d_stl.train_X = d_stl.train_X * 2.0 - 1.0
        d_stl.val_X = d_stl.val_X * 2.0 - 1.0
        d_stl.test_X = d_stl.test_X * 2.0 - 1.0

    print('STL: \ntrain: X.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, \ntest: X.shape={}, y.shape={}'.format(
        d_stl.train_X.shape, d_stl.train_y.shape, d_stl.val_X.shape, d_stl.val_y.shape, d_stl.test_X.shape,
        d_stl.test_y.shape))

    print('STL: train: X.min={}, X.max={}'.format(
        d_stl.train_X.min(), d_stl.train_X.max()))

    d_stl.n_classes = 9

    return d_stl


def load_gtsrb(zero_centre=False, greyscale=False, val=False):
    # Load syn digits
    print('Loading GTSRB...')
    if val:
        d_gts = Dataset.GTSRB(n_val=10000)
    else:
        d_gts = Dataset.GTSRB(n_val=0)

    d_gts.train_X = d_gts.train_X[:]
    d_gts.val_X = d_gts.val_X[:]
    d_gts.test_X = d_gts.test_X[:]
    d_gts.train_y = d_gts.train_y[:]
    d_gts.val_y = d_gts.val_y[:]
    d_gts.test_y = d_gts.test_y[:]

    if greyscale:
        d_gts.train_X = rgb2grey_tensor(d_gts.train_X)
        d_gts.val_X = rgb2grey_tensor(d_gts.val_X)
        d_gts.test_X = rgb2grey_tensor(d_gts.test_X)

    if zero_centre:
        d_gts.train_X = d_gts.train_X * 2.0 - 1.0
        d_gts.val_X = d_gts.val_X * 2.0 - 1.0
        d_gts.test_X = d_gts.test_X * 2.0 - 1.0

    print('GTSRB: \ntrain: X.shape={}, y.shape={}, \nval: X.shape={}, y.shape={}, '
          '\ntest: X.shape={}, y.shape={}'.format(
        d_gts.train_X.shape, d_gts.train_y.shape, d_gts.val_X.shape, d_gts.val_y.shape, d_gts.test_X.shape,
        d_gts.test_y.shape))

    print('GTSRB: train: X.min={}, X.max={}'.format(
        d_gts.train_X.min(), d_gts.train_X.max()))

    d_gts.n_classes = 43

    return d_gts


def load_gtsrb_32(path='../data/data_gtsrb'):
    data_target = pkl.load(open(path), encoding='bytes')
    target_train = np.random.permutation(len(data_target['image']))
    data_t_im = data_target['image'][target_train[:31367], :, :, :]
    data_t_im_test = data_target['image'][target_train[31367:], :, :, :]
    data_t_label = data_target['label'][target_train[:31367]] + 1
    data_t_label_test = data_target['label'][target_train[31367:]] + 1
    data_t_im = data_t_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_t_im_test = data_t_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    return data_t_im, data_t_label, data_t_im_test, data_t_label_test


if __name__ == '__main__':
    USPS = load_USPS(root_dir='../data/usps-dataset/')
    data_loader = torch.utils.data.DataLoader(
        USPS['train'],
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    print(iter(data_loader).next()[0].size())
