import os
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

import pandas as pd
import numpy as np
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils


class DataGen(Dataset):

    def __init__(self, data, transform=True, config_path='config.json'):
        self.config = utils.Configuration(config_path)
        self.data = data
        self.transform = transform
        self.dataset_list = [
                            'data_tmin.csv', 'data_tmax.csv',  # Temperature-related data
                            'data_vap.csv', 'data_pet.csv', 'data_vpd.csv',  # Humidity and evaporation-related data
                            'data_srad.csv', 'data_ws.csv',  # Radiation and weather-related data
                            'data_ppt.csv', 'data_q.csv', 'data_swe.csv',  # Precipitation and water-related data
                            'data_def.csv', 'data_aet.csv', 'data_PDSI.csv', 'data_soil.csv'  # Soil and environmental stress data
                            ]

        # self.aet_dataset = pd.read_csv(os.path.join(self.config.datapath,'data_aet.csv'))
        # self.dataset = self.dataset.astype(float) 
        self.datasets = self.load_datasets()

    def load_datasets(self):
        datasets = {}
        for dataset_name in self.dataset_list:
            dataset_path = os.path.join(self.config.datapath, dataset_name)
            datasets[dataset_name] = pd.read_csv(dataset_path)
        return datasets


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # print(idx)
        all_data = []

        for dataset_name in self.dataset_list:
            dataset = self.datasets[dataset_name]
            data_ = dataset[dataset['cell-id'] == self.data[idx]].iloc[:, -12:]
            data_ = np.array(data_.astype(float)).flatten()
            data_ = np.nan_to_num(data_)
            all_data.append(data_)

        # Stack all data to shape (14, 12)
        data_ = np.stack(all_data, axis=0)
        # print(data_.shape)
        data = torch.from_numpy(data_).float()

        label = data.clone().detach()
        # print(data)

        if self.transform:
            data = self.apply_combined_noise(data)

        return data, label
    
    def normalize_data(self, data):
        pass

    def add_gaussian_noise(self, data, mean=0.0, std_dev=1.0):
        noise = torch.randn(data.size()) * std_dev + mean
        noisy_data = data + noise
        return noisy_data

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

    df = pd.read_csv(os.path.join(config.datapath,'data_aet.csv'))
    ids = list(df['cell-id'])
    # print(ids)
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
        inputs, label = data
        print(inputs.shape)
        print(label.shape)
        print(f'{i} ***************')
        
        # if i==2:
        #     break

    