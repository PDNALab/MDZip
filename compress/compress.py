import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AE(nn.Module):
    def __init__(self, n_atoms, n_channels=4096, latent_dim=20):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            ##(N,1,n_atoms,03)
            nn.Conv2d(1,n_channels,kernel_size=(n_atoms,1), bias=True), #(N,4096,1,3)
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(n_channels),
            
            nn.Conv2d(n_channels,n_channels//4,kernel_size=(1,3), bias=True), #(N,1024,1,1)
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(n_channels//4),
            
            nn.Flatten(),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            #(N,latent_dim,1,1)
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Unflatten(1,(1024,1,1)),
            
            nn.ConvTranspose2d(1024,n_channels,kernel_size=(1,3), bias=True), #(N,4096,1,3)
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(n_channels, 1, kernel_size=(n_atoms,1), bias=True) #(N, 1, n_atoms, 3)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class RMSDLoss(nn.Module):
    def __init__(self):
        super(RMSDLoss, self).__init__()

    def forward(self, recon, x):
        rmsd = torch.sqrt(torch.mean((recon - x) ** 2))
        return rmsd