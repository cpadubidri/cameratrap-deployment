from ae_resnet34 import Autoencoder
from datagen import dataloader
import torch.nn as nn
import torch.optim as optim
import torch

def train():
    #Define model
    model = Autoencoder()


    #load data
    config_path = 'assets/config/config.json'
    train_loader, test_loader = dataloader(config_path)


    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        #train loop
        train_loss = 0.0
        for images, _ in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(images)
            
            # Compute loss
            loss = criterion(outputs, images)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        
        #test loop
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                
                # Forward pass
                outputs, _ = model(images)
                
                # Compute loss
                loss = criterion(outputs, images)
                
                valid_loss += loss.item()
        
        avg_test_loss = valid_loss/len(test_loader)

        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')



if __name__=="__main__":
    train()