import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import pickle
import numpy
import argparse
import os
import shutil
import platform

import sys
sys.path.append('../')
from molzip.autoencoder import *
from molzip.utils import *
from molzip.molzip import *

set_seed()

# Inputs ----------------
parser = argparse.ArgumentParser(description="Compress MD trjectories")

parser.add_argument('-t', '--traj', type=str, required=True, help='Path to the trajectory file [netCDF|XYZ|XTC]')
parser.add_argument('-p', '--top', type=str, required=True, help='Path to the topology file')
parser.add_argument('-s', '--stride', type=int, required=False, help='Read every strid-th frame [Default=1]', default=1)
parser.add_argument('-o', '--out', type=str, required=False, help='Path to save compressed files [Default=current directory]', default=os.getcwd())
parser.add_argument('-n', '--preFix', type=str, required=False, help='Prefix for all generated files [Default=None]', default='')
parser.add_argument('-e', '--epochs', type=int, required=False, help='Number of epochs to train AE model [Default=100]', default=100)
parser.add_argument('-b', '--batchSize', type=int, required=False, help='Batch size to train AE model [Default=128]', default=128)
parser.add_argument('-l', '--latent', type=int, required=False, help='Latent vector length [Default=20]', default=20)
parser.add_argument('-m', '--memmap', type=int, required=False, help='Use memory-map to read trajectory (0: Don\'t use memmap, 1: Use memmap)', default=False)

args = parser.parse_args()
traj = args.traj
top = args.top
stride = args.stride
out = args.out
f_name = args.preFix
epochs = args.epochs
batchSize = args.batchSize
lat = args.latent
memmap = args.memmap
del args

train(traj=traj, top=top, out=out, fname=f_name, epochs=epochs, lat=lat)

# cont_train(traj=traj, top=top,  model=out+f_name+'_compressed/'+f_name+'_model.pt',
#            checkpoint=out+f_name+'_compressed/'+f_name+'_checkpoints.pt', 
#            cluster=out+f_name+'_compressed/'+f_name+'_clusters.pkl', epochs=epochs)

compress(
    traj=traj, top=top, model=out+f_name+'_compressed/'+f_name+'_model.pt',
    cluster=out+f_name+'_compressed/'+f_name+'_clusters.pkl',
    out=out+f_name+'_compressed/'
         )

shutil.copy(top, dst=out+f_name+'_compressed/')

decompress(
    top=top, model=out+f_name+'_compressed/'+f_name+'_model.pt', 
    cluster=out+f_name+'_compressed/'+f_name+'_clusters.pkl', 
    out=out+f_name+f'_compressed/recon_L{lat}.nc',
    compressed=out+f_name+'_compressed/'+f_name+'_compressed.pkl')

