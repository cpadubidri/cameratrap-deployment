from dataGenImg import dataloader
from dae_model import ConvDAE

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils

from torch.optim import Adam
from torch.nn import MSELoss
import torch
from tqdm import tqdm

def train(config):
    #load data
    train_loader, test_loader = dataloader('config.json')

    #load model
    condae = ConvDAE()

    #parameters
    criterion = MSELoss()
    optimizer = Adam(condae.parameters(), lr=0.001)

    #log
    savepath = f'./train-results/{config.trainid}'
    if not(os.path.exists(savepath)):
        os.mkdir(savepath)

    Losslogger = utils.starlogger(name='Loss_logger', filename=f'./{savepath}/trainloss.log',console=False)

    for epoch in range(config.epoch):
        condae.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            outputs, _ = condae(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation loop
        condae.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader):
                outputs, _ = condae(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
        
        test_loss /= len(test_loader.dataset)
        
        print(f'Epoch {epoch+1}/{config.epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        Losslogger.debug(f'Epoch {epoch+1}/{config.epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if epoch%config.savemodelstep==0:
                torch.save(condae.state_dict(), os.path.join(savepath,f'epoch_{epoch}_model.pth'))
    
    
    
    torch.save(condae.state_dict(), os.path.join(savepath, config.trainid+'_final.pth'))

if __name__=='__main__':
    config = utils.Configuration('config.json')
    train(config)