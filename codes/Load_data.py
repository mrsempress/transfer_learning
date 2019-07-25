# encoding=utf-8
"""
    Created on 13:18 2019/07/20
    @author: Chenxi Huang
    It is the auxiliary of fine-tune used to load Office-31 data. And in order to use the Pretrained model, it should
    normalize first.
"""
import torch
from torchvision import datasets, transforms
import Data_transform
import Dataset


def transform_for_Digits(resize_size, Gray_to_RGB=False):
    """
    transform for office
    :param resize_size:
    :param Gray_to_RGB:
    :return:
    """
    T = {
        'train': [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ],
        'test': [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ]
    }

    if Gray_to_RGB:
        for phase in T:
            T[phase].append(transforms.Lambda(lambda x: x.expand([3, -1, -1]).clone()))

    for phase in T:
        T[phase] = transforms.Compose(T[phase])

    return T


def load_data(root_path, dir, batch_size, phase):
    """
    transform for Office.
    make it normalized
    ToTensor(): make it in range 0 and 1
    Normaliza(mean, std): channel =（channel - mean）/ std
    :param root_path:
    :param dir:
    :param batch_size:
    :param phase:
    :return:
    """
    transform_dict = {
        'src': transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'tar': transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}
    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict[phase])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    return data_loader


def load_train(root_path, dir, batch_size, phase):
    """
    Load data for train set
    :param root_path:
    :param dir:
    :param batch_size:
    :param phase:
    :return:
    """
    transform_dict = {
        'src': transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'tar': transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}
    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict[phase])
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=False,
                                             num_workers=4)
    return train_loader, val_loader


class JointDataset(torch.utils.data.Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return min([len(d) for d in self.datasets])

    def __getitem__(self, index):
        return [ds[index] for ds in self.datasets]


def load_dataset(path, train=True):
    """
    load mnist and svhn dataset
    :param path:
    :param train:
    :return:
    """
    img_size = 32

    transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    mnist = datasets.MNIST(path, train=train, download=True, transform=transform)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.188508, 0.19058265, 0.18615675))
    ])
    svhn = datasets.SVHN(path, split='train' if train else 'test', download=True, transform=transform)

    return {'mnist': mnist, 'svhn': svhn}


class BaseDataLoader:
    def __init__(self):
        pass

    def initialize(self, batch_size):
        self.batch_size = batch_size
        self.serial_batches = 0
        self.nThreads = 2
        self.max_dataset_size = float("inf")
        pass

    def load_data(self):
        return None


def return_dataset(data, scale=False, usps=False, all_use='no'):
    """
    return the choose dataset
    :param data:
    :param scale:
    :param usps:
    :param all_use:
    :return:
    """
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = Data_transform.load_svhn_32()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = Data_transform.load_mnist_32(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = Data_transform.load_usps_32(all_use=all_use)
    if data == 'synth':
        train_image, train_label, \
        test_image, test_label = Data_transform.load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, \
        test_image, test_label = Data_transform.load_gtsrb_32()

    return train_image, train_label, test_image, test_label


def dataset_read(source, target, batch_size, scale=False, all_use='no'):
    """
    read the dataset
    :param source:
    :param target:
    :param batch_size:
    :param scale:
    :param all_use:
    :return:
    """
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    usps = False
    if source == 'usps' or target == 'usps':
        usps = True

    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,
                                                                            usps=usps, all_use=all_use)
    train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, usps=usps,
                                                                            all_use=all_use)

    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train

    # input target samples for both
    S_test['imgs'] = test_target
    S_test['labels'] = t_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test
    scale = 40 if source == 'synth' else 28 if source == 'usps' or target == 'usps' else 32
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()
    return dataset, dataset_test


class PairedData(object):
    """
    make pair of data
    """
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': A, 'S_label': A_paths,
                    'T': B, 'T_label': B_paths}


class UnalignedDataLoader:
    """
    make it aligned
    """
    def initialize(self, source, target, batch_size1, batch_size2, scale=32):
        transform = transforms.Compose([
            transforms.Scale(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_source = Dataset.Dataset(source['imgs'], source['labels'], transform=transform)
        dataset_target = Dataset.Dataset(target['imgs'], target['labels'], transform=transform)
        # dataset_source = tnt.dataset.TensorDataset([source['imgs'], source['labels']])
        # dataset_target = tnt.dataset.TensorDataset([target['imgs'], target['labels']])
        data_loader_s = torch.utils.data.DataLoader(
            dataset_source,
            batch_size=batch_size1,
            shuffle=True,
            num_workers=4)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=batch_size2,
            shuffle=True,
            num_workers=4)
        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s, data_loader_t,
                                      float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))
