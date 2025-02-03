import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import itertools
from .utils import *

class Loss(nn.Module):
    r'''
Loss function
    '''
    def __init__(self):
        super(Loss, self).__init__()
        
    def dist_mat(self, coords):
        return(torch.cdist(coords,coords))
    
    def compute_angles(self, tensor):
        num_atoms = tensor.shape[2]
        all_indices = torch.arange(num_atoms, device=tensor.device)
        
        # Generate unique triplets of indices
        triplets = list(itertools.combinations(all_indices, 3))
        triplets = torch.tensor(triplets, dtype=torch.long, device=tensor.device)

        # Get indices for each triplet
        i = triplets[:, 0]
        j = triplets[:, 1]
        k = triplets[:, 2]
        
        a = tensor[:, 0, i, :]
        b = tensor[:, 0, j, :]
        c = tensor[:, 0, k, :]

        # Efficient vector calculations
        ba = b - a
        bc = b - c
        
        ba_norm = ba.norm(dim=-1, keepdim=True)
        bc_norm = bc.norm(dim=-1, keepdim=True)

        # Normalize vectors
        ba_normalized = ba / ba_norm
        bc_normalized = bc / bc_norm
        
        # Calculate cosines of angles
        cos_angles = (ba_normalized * bc_normalized).sum(dim=-1)
        angles = torch.acos(cos_angles.clamp(-1.0, 1.0))
        
        return angles

    def minMax_scale(self, tensor1, tensor2):
        stacked_tensor = torch.stack([tensor1, tensor2])
        min_val = torch.min(stacked_tensor).item()
        max_val = torch.max(stacked_tensor).item()
        tensor1 =  (tensor1 - min_val) / (max_val - min_val)
        tensor2 =  (tensor2 - min_val) / (max_val - min_val)
        return tensor1, tensor2
        
    def forward(self, recon, x, w:float=1.0, a:float=1.0):
        torch.cuda.empty_cache()  # Clear the cache before computations
        coord1, coord2 = self.minMax_scale(recon, x)
        rmse = torch.sqrt(torch.mean((coord1 - coord2) ** 2))
        if w > 1e-4:
            dist1, dist2 = self.minMax_scale(self.dist_mat(recon), self.dist_mat(x))
            rmse_d = torch.sqrt(torch.mean((dist1 - dist2) ** 2))
        else:
            rmse_d = 0.0
        if a > 1e-4:
            ang1, ang2 = self.minMax_scale(self.compute_angles(recon), self.compute_angles(x))
            rmse_a = torch.sqrt(torch.mean((ang1 - ang2) ** 2))
        else:
            rmse_a = 0.0
        return rmse + w*(rmse_d) + a*(rmse_a)

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
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(n_channels,n_channels//4,kernel_size=(1,3), bias=True), #(N,1024,1,1)
            nn.BatchNorm2d(n_channels//4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
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
    def __init__(self, model, lr=1e-4, idx=None, w:float=1.0, a:float=1.0, loss_path:str=os.getcwd()+"losses.dat", epoch_losses=None):
        r'''
pytorch-Lightning AutoEncoder
-----------------------------
model : pytorch model
lr (float, Tensor, optional) : learning rate [default: 1e-4]
weight_decay (float, optional) : weight decay (L2 penalty) [default=0]
idx : a list of indices or a list of tuples of indices that specify a subset of the data x and recon to be used in the loss calculation. [Default=None]
When idx is not None, it is assumed to be a list of tuples, where each tuple contains two indices i[0] and i[1]. These indices are used to slice the data x and recon along the first axis (i.e., x[:, i[0]:i[1]] and recon[:, i[0]:i[1]]). The loss is then calculated for each slice separately, and the results are summed up.
loss_path (str) : File path to save losses file [Default=<currunt dir>/losses.dat.
        '''
        super().__init__()
        self.model = model
        self.w = w
        self.a = a
        self.loss_fn = Loss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.idx = idx
        self.training_losses = []
        self.epoch_losses = epoch_losses if epoch_losses is not None else []
        self.loss_path = loss_path

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, x, recon, idx):
        if idx is None:
            return self.loss_fn(recon, x, w=self.w, a=self.a)
        else:
            return sum(self.loss_fn(x[:, i[0]:i[1]], recon[:, i[0]:i[1]], w=self.w, a=self.a) for i in idx)

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


