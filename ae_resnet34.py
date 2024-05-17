import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        resnet34 = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        self.encoder = nn.Sequential(
            resnet34.conv1,   # Output size: 256x256
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool, # Output size: 128x128
            resnet34.layer1,  # Output size: 128x128
            resnet34.layer2,  # Output size: 64x64
            resnet34.layer3,  # Output size: 32x32
            resnet34.layer4   # Output size: 16x16
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),   # 128x128 -> 256x256
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 256x256 -> 512x512
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),    # 512x512 -> 512x512 
            nn.Sigmoid()  
        )
        
    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = ResNet34Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        latent_vec = self.encoder(x)

        x = self.decoder(latent_vec)
        return x, F.normalize(latent_vec, dim=-1)


if __name__=='__main__':
    
    autoencoder = Autoencoder()
    input_image = torch.randn((1, 3, 512, 512))

    output_image, latent_vec = autoencoder(input_image)

    print(output_image.shape)
    print(latent_vec.shape) 
