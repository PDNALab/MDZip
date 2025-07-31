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
        
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, n_atoms:int, n_channels:int=4096, latent_dim:int=20):
        super().__init__()
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim

        if self.n_atoms*3 < 1024:
            self.n_channels = 1024
        elif self.n_atoms*3 < 2048:
            self.n_channels = 2048
        else:
            self.n_channels = n_channels

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, self.n_channels, kernel_size=(n_atoms,1), bias=True),
            nn.BatchNorm2d(self.n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.n_channels, self.n_channels//2, kernel_size=(1,3), bias=True),
            nn.BatchNorm2d(self.n_channels//2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoderSkip1 = nn.Sequential(
            nn.Conv2d(1, self.n_channels//2, kernel_size=(n_atoms,3), bias=True),
            nn.BatchNorm2d(self.n_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.fuseE1 = nn.Sequential(
            nn.Conv2d(self.n_channels//2 * 2, self.n_channels//2, kernel_size=1),
            nn.BatchNorm2d(self.n_channels//2),
            nn.LeakyReLU(0.2)
        )

        self.encoder2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_channels//2, self.n_channels//4),
            nn.BatchNorm1d(self.n_channels//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.n_channels//4, self.n_channels//8),
            nn.BatchNorm1d(self.n_channels//8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.n_channels//8, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        self.encoderSkip2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_channels//2, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        self.fuseE2 = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        self.encoderReLU = nn.ReLU(inplace=True)

        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, self.n_channels//8),
            nn.BatchNorm1d(self.n_channels//8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.n_channels//8, self.n_channels//4),
            nn.BatchNorm1d(self.n_channels//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.n_channels//4, self.n_channels//2),
            nn.BatchNorm1d(self.n_channels//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (self.n_channels//2,1,1))
        )
        self.decoderSkip1 = nn.Sequential(
            nn.Linear(latent_dim, self.n_channels//2),
            nn.BatchNorm1d(self.n_channels//2),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (self.n_channels//2,1,1))
        )
        
        self.fuseD1 = nn.Sequential(
            nn.Conv2d(self.n_channels//2 * 2, self.n_channels//2, kernel_size=1),
            nn.BatchNorm2d(self.n_channels//2),
            nn.LeakyReLU(0.2)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(self.n_channels//2, self.n_channels, kernel_size=(1,3), bias=True),
            nn.BatchNorm2d(self.n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.n_channels, 1, kernel_size=(n_atoms,1), bias=True)
        )
        self.decoderSkip2 = nn.Sequential(
            nn.ConvTranspose2d(self.n_channels//2, 1, kernel_size=(n_atoms,3), bias=True),
            nn.BatchNorm2d(1)
        )
        
        self.fuseD2 = nn.Conv2d(2, 1, kernel_size=1)

        self.sigAct = nn.Sigmoid()

    def encode(self, x):
        out = self.encoder1(x)
        skip1 = self.encoderSkip1(x)
        
        fused = torch.cat([out, skip1], dim=1)          # (N, 2*(C/2), 1, 1)
        out   = self.fuseE1(fused)                      # back to (N, C/2, 1, 1)

        skip2 = self.encoderSkip2(out)                  # (N, latent)
        out   = self.encoder2(out)                      # (N, latent)
        fused = torch.cat([out, skip2], dim=1)          # (N, 2*latent)
        out   = self.fuseE2(fused)                      # (N, latent)

        return self.encoderReLU(out)

    def decode(self, x):
        skip1 = self.decoderSkip1(x)
        out   = self.decoder1(x)
        fused = torch.cat([out, skip1], dim=1)          # (N, 2*(C/2), 1, 1)
        out   = self.fuseD1(fused)                      # (N, C/2, 1, 1)
        out   = self.encoderReLU(out)

        skip2 = self.decoderSkip2(out)
        out   = self.decoder2(out)
        fused = torch.cat([out, skip2], dim=1)          # (N, 2, H, W)
        out   = self.fuseD2(fused)                      # (N, 1, H, W)

        return self.sigAct(out)

    def forward(self, x):
        y = self.encode(x)
        return self.decode(y)
    
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