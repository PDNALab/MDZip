import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import *


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, recon, x):
        rmse = torch.sqrt(torch.mean((recon - x) ** 2))
        return rmse

class AE(nn.Module):
    def __init__(self, n_atoms:int, latent_dim:int=20, n_channels:int=4096):
        r'''
pytorch-Lightning AutoEncoder
-----------------------------
n_atoms (int) : total number of atoms in a single trajectory frame
latent_dim (int) : compressed latent vector length [Default=20]
        '''
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


class LightAE(pl.LightningModule):
    def __init__(self, model, lr=1e-4, weight_decay=0, idx=None):
        r'''
pytorch-Lightning AutoEncoder
-----------------------------
model : pytorch model
lr (float, Tensor, optional) : learning rate [default: 1e-4]
weight_decay (float, optional) : weight decay (L2 penalty) [default=0]
idx : a list of indices or a list of tuples of indices that specify a subset of the data x and recon to be used in the loss calculation. [Default=None]
When idx is not None, it is assumed to be a list of tuples, where each tuple contains two indices i[0] and i[1]. These indices are used to slice the data x and recon along the first axis (i.e., x[:, i[0]:i[1]] and recon[:, i[0]:i[1]]). The loss is then calculated for each slice separately, and the results are summed up.
        '''
        super().__init__()
        self.model = model
        self.loss_fn = RMSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.idx = idx
        self.training_losses = []
        self.epoch_losses = []

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, x, recon, idx):
        if idx is None:
            return self.loss_fn(recon, x)
        else:
            return sum(self.loss_fn(x[:, i[0]:i[1]], recon[:, i[0]:i[1]]) for i in idx)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        recon = self.model(x)
        loss = self._calculate_loss(x, recon, self.idx)
        self.log('train_loss', loss, on_epoch=True)
        self.training_losses.append(loss.detach().cpu().item())
        return {'loss': loss}

    # def on_train_epoch_end(self):
    #     avg_train_loss = torch.tensor(self.training_losses).mean()
    #     self.log('avg_train_loss', avg_train_loss, prog_bar=True, logger=True)
    #     self.training_losses.clear()
    def on_train_epoch_end(self):
        epoch_loss = torch.tensor(self.training_losses).mean()
        self.log('Epoch Loss', epoch_loss, prog_bar=True, logger=False)
        self.epoch_losses.append(epoch_loss.detach().cpu().item())
        self.training_losses.clear()

    def on_train_end(self):
        print('Autoencoder training complete')
        print('_'*70+'\n')
        with open("losses.dat", "w") as f:
            for i, loss in enumerate(self.epoch_losses):
                f.write(f'{i:4d}\t {loss:8.5f}\n')

    def configure_optimizers(self):
        return self.optimizer


