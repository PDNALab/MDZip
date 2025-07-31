import torch
import torch.nn as nn
import mdtraj as md
import torch.optim as optim
import pytorch_lightning as pl
from openmm.app import *
from openmm import *
from openmm.unit import *
import itertools
from .utils import *

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def forward(self, recon, x):
        return torch.sqrt(torch.mean((recon - x) ** 2))
        
class AE(nn.Module):
    def __init__(self, n_atoms:int, n_channels:int=4096, latent_dim:int=20):
        r'''
pytorch-Lightning AutoEncoder
-----------------------------
n_atoms (int) : total number of atoms in a single trajectory frame
latent_dim (int) : compressed latent vector length [Default=20]
n_channels (int) : number of channels in the first convolution layer [Default=4096]
        '''
        super().__init__()
        
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        
        if self.n_atoms*3 < 1024:
            self.n_channels = 1024
        elif self.n_atoms*3 < 2048:
            self.n_channels = 2048
        else:
            self.n_channels = n_channels
            
        self.encoder = nn.Sequential(
            ##(N,1,n_atoms,03)
            nn.Conv2d(1,n_channels,kernel_size=(n_atoms,1), bias=True), #(N,n_channels,1,3)
            nn.BatchNorm2d(n_channels),
            # nn.Dropout2d(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(n_channels,n_channels//2,kernel_size=(1,3), bias=True), #(N,n_channels/2,1,1)
            nn.BatchNorm2d(n_channels//2),
            # nn.Dropout2d(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(), #(N,n_channels/2)
            
            nn.Linear(n_channels//2, n_channels//4), #(N,n_channels/4)
            nn.BatchNorm1d(n_channels//4),
            # nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_channels//4, n_channels//8), #(N,n_channels/8)
            nn.BatchNorm1d(n_channels//8),
            # nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_channels//8, latent_dim), #(N,latent_dim)
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
            
        self.decoder = nn.Sequential(
            #(N,latent_dim)
            nn.Linear(latent_dim, n_channels//8), #(N,n_channels/8)
            nn.BatchNorm1d(n_channels//8),
            # nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_channels//8, n_channels//4), #(N,n_channels/4)
            nn.BatchNorm1d(n_channels//4),
            # nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(n_channels//4, n_channels//2), #(N,n_channels/2)
            nn.BatchNorm1d(n_channels//2),
            # nn.Dropout(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Unflatten(1,(n_channels//2,1,1)), # (N,n_channels/2,1,1)
        
            nn.ConvTranspose2d(n_channels//2,n_channels,kernel_size=(1,3), bias=True), #(N,n_channels,1,3)
            nn.BatchNorm2d(n_channels),
            # nn.Dropout2d(0.1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(n_channels, 1, kernel_size=(n_atoms,1), bias=True), #(N, 1, n_atoms, 3)
            nn.Sigmoid()  # Sigmoid activation for output
        )
    
    def encode(self, x):
        out = self.encoder(x)
        return out
    
    def decode(self, x):
        out = self.decoder(x)
        return out
    
    def forward(self, x):
        # Encoder
        out = self.encode(x)
        # Decoder
        out = self.decode(out)
        return out
    
class LightAE(pl.LightningModule):
    def __init__(self, model, lr=1e-4, idx=None, loss_path:str=os.getcwd()+"losses.dat", epoch_losses=None, weight_decay=0.01):
        r'''
pytorch-Lightning AutoEncoder
-----------------------------
model : pytorch model
lr (float, Tensor, optional) : learning rate [default: 1e-4]
weight_decay (float, optional) : weight decay for the optimizer [default: 0.01]
idx : a list of indices or a list of tuples of indices that specify a subset of the data x and recon to be used in the loss calculation. [Default=None]
When idx is not None, it is assumed to be a list of tuples, where each tuple contains two indices i[0] and i[1]. These indices are used to slice the data x and recon along the first axis (i.e., x[:, i[0]:i[1]] and recon[:, i[0]:i[1]]). The loss is then calculated for each slice separately, and the results are summed up.
loss_path (str) : File path to save losses file [Default=<currunt dir>/losses.dat.
epoch_losses (list, optional) : List to store epoch losses. If None, a new list is created. [Default=None]
        '''
        super().__init__()
        self.model = model
        self.loss_fn = Loss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.idx = idx
        self.training_losses = []
        self.epoch_losses = epoch_losses if epoch_losses is not None else []
        self.loss_path = loss_path
        
    def forward(self, x):
        return self.model(x)
    
    def _calculate_loss(self, x, recon, idx):
        if idx is None:
            return self.loss_fn(recon, x)
        else:
            return sum(self.loss_fn(x[:, i[0]:i[1]], recon[:, i[0]:i[1]]) for i in idx)

    def training_step(self, batch):
        x = batch.to(self.device)
        recon = self.model(x)
        loss = self._calculate_loss(x, recon, self.idx)
        self.log('train_loss', loss, on_epoch=True)
        self.training_losses.append(loss.detach().cpu().item())
        return {'loss': loss}

    def on_train_epoch_end(self):
        epoch_loss = torch.tensor(self.training_losses).mean()
        self.log('Epoch Loss', epoch_loss, prog_bar=True, logger=False)
        self.epoch_losses.append(epoch_loss.detach().cpu().item())
        self.training_losses.clear()

    def on_train_end(self):
        print('Autoencoder training complete')
        print('_'*70+'\n')
        with open(self.loss_path, "w") as f:
            for i, loss in enumerate(self.epoch_losses):
                f.write(f'{i:>8d}\t {loss:8.5f}\n')

    def configure_optimizers(self):
        return self.optimizer