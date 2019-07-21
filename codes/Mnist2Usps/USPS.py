# encoding=utf-8
"""
    Created on 09:02 2019/07/21
    @author: Chenxi Huang
    This is transform the data, so the train and test data can be same size.
"""
import numpy as np
from torch.utils import data
import h5py
import os
from torchvision import transforms
import torch


class USPSDataset(data.Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.target_transform = transforms.ToTensor()
        with h5py.File(os.path.join(root_dir,'usps.h5'), 'r') as hf:
            if train:
                d = hf.get('train')
            else:
                d = hf.get('test')

            # format:(7291, 256)
            self.samples = d.get('data')[:]

            # format:(7291,)
            self.labels = d.get('target')[:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.samples[index]
        img = img.reshape(16, 16)
        img = img[:, :, np.newaxis]
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[index],dtype=torch.long)
        return [img, label]
