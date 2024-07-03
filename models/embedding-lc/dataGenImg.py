import os
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

import pandas as pd
import numpy as np
import random
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils


class DataGen(Dataset):

    def __init__(self, ids, transform=None, config_path='config.json'):
        self.config = utils.Configuration(config_path)
        self.ids = ids
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: self.min_max_normalize(x))
            
        ])

    def __len__(self):
        return len(self.ids)
    

    def __getitem__(self, idx):
        # print(idx)
        image = cv2.imread(os.path.join(self.config.datapath, self.ids[idx]))        

        if self.transforms:
            image = self.transforms(image[:,:,0])     
        

        data = image
        label = image
        data = self.add_gaussian_noise(data)
        return data, label, self.ids[idx]
    
    def min_max_normalize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val != min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        else:
            tensor = torch.zeros_like(tensor)
        return tensor

    def add_gaussian_noise(self, data, mean=0.0, std_dev=0.25):
        gaus_data = np.random.normal(mean, std_dev, data.shape)+data.numpy()
        gaus_data = np.array(gaus_data, dtype=np.float32)
        # print(gaus_data.dtype)
        # print(data.shape)
        return gaus_data

    def add_batchswap_noise(self, batch, swap_prob=0.1):
        batch_size, num_features = batch.shape
        for i in range(batch_size):
            if np.random.rand() < swap_prob:
                j = np.random.randint(0, batch_size)
                batch[i], batch[j] = batch[j].clone(), batch[i].clone()
        return batch

    def apply_combined_noise(self, data, gaussian_mean=0.0, gaussian_std_dev=0.25, swap_prob=0.1):
        data_with_gaussian_noise = self.add_gaussian_noise(data, gaussian_mean, gaussian_std_dev)
        noisy_data = self.add_batchswap_noise(data_with_gaussian_noise, swap_prob)
        return noisy_data

        
        


def dataloader(config_path):
    config = utils.Configuration(config_path)

    ids = os.listdir(config.datapath)
    random.seed(40)
    random.shuffle(ids)
    # print(ids)
    split_idx = len(ids)-int(len(ids)*config.testrain_split)
    train_ids, test_ids = ids[:split_idx], ids[split_idx:]
    # print(test_ids)

    train_dataset = DataGen(train_ids,  config_path=config_path)
    test_dataset = DataGen(test_ids, config_path=config_path)
    

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    
    return train_loader, test_loader






if __name__=='__main__':
    train_loader, test_loader = dataloader('config.json')

    for i, data in enumerate(train_loader):
        inputs, label, _ = data
        print(label.shape)
        ex = np.moveaxis((inputs[0].numpy()*255), 0, 2)
        ex_l = np.moveaxis((label[0].numpy()*255), 0, 2)
        # print(ex.shape)
        cv2.imwrite('test.png',ex)
        cv2.imwrite('test1.png',ex_l)
        print(f'{i} ***************')
        
        if i==0:
            break

    