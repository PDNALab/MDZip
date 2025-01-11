import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle
import numpy
import argparse
import os
import shutil

from .utils import *
from .autoencoder import *

torch.set_float32_matmul_precision('medium')

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
parser.add_argument('-m', '--memmap', type=int, required=False, help='Use memory-map to read trajectory (0: Don\'t use memmap, 1: Use memmap) [Default=0]', default=0)

args = parser.parse_args()
traj = args.traj
top = args.top
stride = args.stride
out = args.out
fname = args.preFix
epochs = args.epochs
batchSize = args.batchSize
lat = args.latent
memmap = args.memmap
del args

pathExists(traj)
pathExists(top)
pathExists(out)

if not out.endswith('/'):
    out += '/'

if memmap not in [1,0]:
    raise ValueError('memmap can have either 0 or 1 only') 
else:
    memmap = True if memmap==1 else False
    
# Define device ---------
if torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = 'gpu'
    n_devices = 1
    print('Device name:', torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    accelerator = 'cpu'
    n_devices = None
    print('Device name: CPU')

# Read trajectory -------
traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap) # idea is to use all available data to train model
n_atoms = traj_.shape[2]
traj_dl = DataLoader(traj_, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=4)
print('_'*70+'\n')

# Train model -----------
model = AE(n_atoms=n_atoms, latent_dim=lat)
model = LightAE(model=model, lr=1e-4, weight_decay=0)

print('Training Deep Convolutional AutoEncoder model')

trainer = pl.Trainer(max_epochs=epochs, accelerator=accelerator, devices=n_devices)
trainer.fit(model, traj_dl)

with open(out+fname+'_trainingLoss.dat', 'w') as fl:
    fl.write(f"{'#epoch':>8}\t{'RMSE (nm)':>10}")
    for i, loss in enumerate(model.epoch_losses):
        fl.write(f"{i:>8d}{loss:8.3f}")

rmsd, r2, mean_squared_error = fitMetrics(model=model, dl=traj_dl, top=top)
print('\n')

# Compress --------------
torch.save(model, out+fname+'model.pt')

encoder = model.model.encoder.to(device)
traj_dl = DataLoader(traj_, batch_size=batchSize, shuffle=False, drop_last=False, num_workers=4) 

z = []
with torch.no_grad():
    encoder.eval()
    for batch in tqdm(traj_dl, desc="Compressing "):
        batch = batch.to(device=device, dtype=torch.float32)
        z.append(encoder(batch))

pickle.dump(z, open(out+fname+"_compressed.pkl", 'wb'))
print('_'*70+'\n')

# Print stats -----------
org_size = os.path.getsize(traj)
comp_size = os.path.getsize(out+fname+"_compressed.pkl")
compression = 100*(1 - comp_size/org_size)

template = "{string:<20} :{value:15.3f}"
print(template.format(string='Original Size [MB]', value=round(org_size*1e-6,3)))
print(template.format(string='Compressed Size [MB]', value=round(comp_size*1e-6,3)))
print(template.format(string='Compression %', value=round(compression,3)))
print('---')
print(template.format(string='RMSD [\u212B]', value=round(np.mean(rmsd),3)))
print(template.format(string='R\u00b2', value=round(r2,3))) 
print(template.format(string='MSE (nm\u00b2)', value=round(mean_squared_error,3)))

# Clean -----------------
if os.path.exists('lightning_logs'):
    shutil.rmtree('lightning_logs')
if os.path.exists('temp_traj.dat'):
    os.remove('temp_traj.dat')

# End -------------------
print('\n')