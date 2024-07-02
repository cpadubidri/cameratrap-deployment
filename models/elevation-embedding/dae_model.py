import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        # self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 512)

        self.encoder[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(512, 512 * 16 * 16)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 16, 16)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = torch.sigmoid(self.upconv4(x))
        return x

class ConvDAE(nn.Module):
    def __init__(self):
        super(ConvDAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_space = self.encoder(x)
        x = self.decoder(latent_space)
        return x, latent_space

# Example usage
if __name__ == "__main__":
    model = ConvDAE()
    input_image = torch.randn((1, 1, 256, 256))  # Example input (batch size, channels, height, width)
    output_image, emb = model(input_image)
    print(output_image.shape)  
    print(emb.shape)
