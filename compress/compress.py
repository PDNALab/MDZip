import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class convVAE(nn.Module):
    def __init__(self, n_channels=4096, n_atoms = 127, latent_dim=16):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            ##(N,1,n_atoms,03)
            nn.Conv2d(1,n_channels,kernel_size=(n_atoms,1),stride=3,padding=3), #(N,4096,3,3)
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels,n_channels//4,kernel_size=(3,1)), #(N,1024,1,3)
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels//4,n_channels//16,kernel_size=(1,1)), #(N,256,1,3)
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channels//16,latent_dim,kernel_size=(1,3)) #(N,latent_dim,1,1)
        )
        
        self.decoder = nn.Sequential(
            #(N,latent_dim,1,1)
            nn.ConvTranspose2d(latent_dim,n_channels//16,kernel_size=(1,3)), #(N,256,1,3)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_channels//16, n_channels//4, kernel_size=(1,1)), #(N,1024,1,3)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_channels//4, n_channels, kernel_size=(1,1)), #(N,4096,1,3)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_channels, 1, kernel_size=(n_atoms,1)), #(N, 1, n_atoms, 3)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # symmetric_output = (decoded + decoded.transpose(-1, -2)) / 2.0  # Calculate symmetric output
#         clamped_output = torch.clamp(symmetric_output, min=0, max=1.0)
        return decoded