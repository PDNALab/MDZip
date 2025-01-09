import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
import numpy
import argparse
import os

from utils import *
from autoencoder import *

torch.set_float32_matmul_precision('medium')

# Inputs ----------------
parser = argparse.ArgumentParser(description="Compress MD trjectories")

parser.add_argument('-t', '--traj', type=str, required=True, help='Path to the trajectory file [netCDF|XYZ|XTC]')
parser.add_argument('-p', '--top', type=str, required=True, help='Path to the topology file')
parser.add_argument('-s', '--stride', type=int, required=False, help='Read every strid-th frame [Default=1]', default=1)
parser.add_argument('-o', '--out', type=str, required=False, help='Path to save compressed files [Default=current directory]', default=os.getcwd())
parser.add_argument('-e', '--epochs', type=int, required=False, help='Number of epochs to train AE model [Default=100]', default=100)
parser.add_argument('-b', '--batchSize', type=int, required=False, help='Batch size to train AE model [Default=128]', default=128)
parser.add_argument('-l', '--latent', type=int, required=False, help='Latent vector length [Default=20]', default=20)
parser.add_argument('-m', '--memmap', type=int, required=False, help='Use memory-map to read trajectory (0: Don\'t use memmap, 1: Use memmap) [Default=0]', default=0)

args = parser.parse_args()
traj = args.traj
top = args.top
stride = args.stride
out = args.out
epochs = args.epochs
batchSize = args.batchSize
lat = args.latent
memmap = args.memmap
del args

pathExists(traj)
pathExists(top)
pathExists(out)

if memmap not in [1,0]:
    raise ValueError('memmap can have either 0 or 1 only') 
else:
    memmap=True if memmap==1 else memmap=False
    
# Define device --------
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Device name:', torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print('Device name: CPU')

# Read trajectory -----
traj = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)
n_atoms = traj.shape[2]
traj_dl = DataLoader(traj, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=4)
print('_'*70+'\n')

# Train model ---------
model = AE(n_atoms=n_atoms, latent_dim=lat)
model = LightAE(model=model, lr=1e-4, weight_decay=0)

print('Training Deep Convolutional AutoEncoder model')
