import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision import transforms
import random
from utils import Configuration
import pandas as pd
import ast
from torch.utils.data._utils.collate import default_collate
import numpy as np
import matplotlib.pyplot as plt


class DataGen(Dataset):
    '''
    This is a class to feed data during training
    
    '''
    def __init__(self, train_ids, config_path='assets/config/config.json'):
        
        self.config = Configuration(config_path)
        self.ids = train_ids
        

        self.transform_data = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
        

    
    def __getitem__(self, idx):
        # Create image path for the given id (idx)
        img_path = os.path.join(self.config.datapath, self.ids[idx])
        
        # Read the image file
        image = self.__readfile__(img_path)
        # if image is None:
        #     raise ValueError(f"Image at path {img_path} could not be loaded.")
        
        # Apply transformations to the image
        if self.transform_data:
            image = self.transform_data(image)

        # # Create a dummy label for demonstration purposes (you should replace this with actual labels)
        # label = torch.tensor(0)

        return image, image
        

    def __readfile__(self, img_path):
        # Load and resize the image according to the configuration.
        image = cv2.imread(img_path)
        
        if self.config.image_size!=None:
            image = cv2.resize(image, (self.config.image_size, self.config.image_size))
        return image
            
    def __len__(self):
        # Return the total number of items in the dataset.
        return len(self.ids)

def dataloader(config_path):
    """
    Creates data loaders for training and validation/testing. Configurations are read from a JSON file.
    """
    config = Configuration(config_path) #change configuration in config.json

    #list all the available data and split into train and test ids
    ids = config.load_datapath()

    ids = ids[:30]
    random.shuffle(ids)
    split_idx = int(len(ids) * config.testrain_split)
    train_ids, test_ids = ids[split_idx:], ids[:split_idx]

    #train and test datafeeder initialization
    train_dataset = DataGen(train_ids)
    test_dataset = DataGen(test_ids)
    
    #Dataloader class from Pytorch. We can implement this in out Datafeeder but Pytorch provides some GPU parallelizng stuff so lets use this :)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    
    return train_loader, test_loader




if __name__ == '__main__':    
    train_loader, test_loader = dataloader(config_path='assets/config/config.json')
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.shape, labels.shape)

        # Plot the images and labels side by side for a batch of 4
        batch_size = 4
        fig, axs = plt.subplots(batch_size, 2, figsize=(10, 10))
        
        for j in range(batch_size):
            # Get the input and label images for the current index
            input_img = np.transpose(inputs[j].numpy(), (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            label_img = np.transpose(labels[j].numpy(), (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            
            # Plot the input image
            axs[j, 0].imshow(input_img)
            axs[j, 0].set_title("Input")
            axs[j, 0].axis('off')
            
            # Plot the label image
            axs[j, 1].imshow(label_img)
            axs[j, 1].set_title("Label")
            axs[j, 1].axis('off')

        plt.savefig('sample.png')
        break


    