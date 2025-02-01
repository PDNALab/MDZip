import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pickle
import numpy
import argparse
import os
import platform
import shutil
from tqdm import tqdm

from .utils import *
from .autoencoder import *

set_seed()

# -------------------------------------------- 
#             C O M P R E S S                  
# -------------------------------------------- 

def compress(traj:str, top:str, stride:int=1, out:str=os.getcwd(), fname:str='', epochs:int=100, batchSize:int=128, lat:int=20, w:float=1.0, memmap:bool=False):
    r'''
compressing trajectory
----------------------
traj (str) : Path to the trajectory file
top (str) : Path to the topology file
stride (int) : Read every strid-th frame [Default=1]
out (str) : Path to save compressed files [Default=current directory]
fname (str) : Prefix for all generated files [Default=None]
epochs (int) : Number of epochs to train AE model [Default=100]
batchSize (int) : Batch size to train AE model [Default=128]
lat (int) : Latent vector length [Default=20]
w (float) : Non-negative weight for loss function [Default=1.0]
memmap (bool) : Use memory-map to read trajectory [Default=False]
        '''
    
    pathExists(traj)
    pathExists(top)

    if len(fname) != 0:
        fname += '_'
    
    if platform.system() == "Windows":
        if not out.endswith('\\'):
            out += '\\'
    else:
        if not out.endswith('/'):
            out += '/'

    if os.path.exists(out+fname+'compressed'):
        shutil.rmtree(out+fname+'compressed')
    os.mkdir(out+fname+'compressed')
    out = out+fname+'compressed\\' if platform.system() == "Windows" else out+fname+'compressed/'
    
    # Define device ---------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        accelerator = 'gpu'
        n_devices = torch.cuda.device_count()
        if n_devices == 1:
            print('Device name:', torch.cuda.get_device_name(device))
        else:
            print(f'Available devices: {n_devices:>02d}')
            for i in range(n_devices):
                print(torch.cuda.get_device_name(i))
    else:
        device = torch.device('cpu')
        accelerator = 'cpu'
        n_devices = None
        print('CUDA is not available')

    # Read trajectory -------
    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)
    n_atoms = traj_.shape[2]
    traj_dl = DataLoader(traj_, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=4)
    print('_'*70+'\n')

    # Train model -----------
    model = AE(n_atoms=n_atoms, latent_dim=lat)
    model = LightAE(model=model, lr=1e-4, w=w, loss_path=out+fname+'losses.dat')

    print('Training Deep Convolutional AutoEncoder model')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=out,
        filename=f'{fname}checkpoint',
        save_top_k=1  # Save the best checkpoint
        )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=n_devices,
        callbacks=[checkpoint_callback]
        )
    
    trainer.fit(model, traj_dl)

    # rmsd, r2, mean_squared_error = fitMetrics(model=model, dl=traj_dl, top=top)
    print('\n')
    
    torch.save(model, out+fname+'model.pt')

    # Compress --------------
    encoder = model.model.encoder.to(device)
    traj_dl = DataLoader(traj_, batch_size=batchSize, shuffle=False, drop_last=False, num_workers=4) 

    z = []
    with torch.no_grad():
        encoder.eval()
        for batch in tqdm(traj_dl, desc="Compressing "):
            batch = batch.to(device=device, dtype=torch.float32)
            z.append(encoder(batch))
    
    pickle.dump(z, open(out+fname+"compressed.pkl", 'wb'))
    print('_'*70+'\n')

    # Print stats -----------
    org_size = os.path.getsize(traj)
    comp_size = os.path.getsize(out+fname+"compressed.pkl")
    compression = 100*(1 - comp_size/org_size)
    
    template = "{string:<20} :{value:15.3f}"
    print(template.format(string='Original Size [MB]', value=round(org_size*1e-6,3)))
    print(template.format(string='Compressed Size [MB]', value=round(comp_size*1e-6,3)))
    print(template.format(string='Compression %', value=round(compression,3)))
    print('---')
    # print(template.format(string='RMSD [\u212B]', value=round(np.mean(rmsd),3)))
    # print(template.format(string='R\u00b2', value=round(r2,3))) 
    # print(template.format(string='MSE (nm\u00b2)', value=round(mean_squared_error,3)))
    
    # Clean -----------------
    if os.path.exists('lightning_logs'):
        shutil.rmtree('lightning_logs')
    if os.path.exists('temp_traj.dat'):
        os.remove('temp_traj.dat')

    print('\n')
    
# -------------------------------------------- 
#      C O N T I N U E  C O M P R E S S                  
# -------------------------------------------- 

def cont_compress(traj:str, top:str,  model:str, checkpoint:str, stride:int=1, epochs:int=100, batchSize:int=128, lat:int=20, w:float=1.0, memmap:bool=False):
    r'''
compressing trajectory
----------------------
traj (str) : Path to the trajectory file
top (str) : Path to the topology file
stride (int) : Read every strid-th frame [Default=1]
model (str)  = Path to previously trained model file
checkpoint (str) = Path to previously check point file
out (str) : Path to save compressed files [Default=current directory]
fname (str) : Prefix for all generated files [Default=None]
epochs (int) : Number of epochs to train AE model [Default=100]
batchSize (int) : Batch size to train AE model [Default=128]
lat (int) : Latent vector length [Default=20]
w (float) : Non-negative weight for loss function [Default=1.0]
memmap (bool) : Use memory-map to read trajectory [Default=False]
        '''
    
    pathExists(traj)
    pathExists(top)

    # Load model
    model = torch.load(model).to(device)
    
    out = os.path.dirname(model.loss_path)
    fname = os.path.basename(model.loss_path).split('_')[0]
    
    if len(fname) != 0:
        fname += '_'
    
    if platform.system() == "Windows":
        if not out.endswith('\\'):
            out += '\\'
    else:
        if not out.endswith('/'):
            out += '/'

    if os.path.exists(out+fname+'compressed'):
        shutil.rmtree(out+fname+'compressed')
    os.mkdir(out+fname+'compressed')
    out = out+fname+'compressed\\' if platform.system() == "Windows" else out+fname+'compressed/'
    
    # Define device ---------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        accelerator = 'gpu'
        n_devices = torch.cuda.device_count()
        if n_devices == 1:
            print('Device name:', torch.cuda.get_device_name(device))
        else:
            print(f'Available devices: {n_devices:>02d}')
            for i in range(n_devices):
                print(torch.cuda.get_device_name(i))
    else:
        device = torch.device('cpu')
        accelerator = 'cpu'
        n_devices = None
        print('CUDA is not available')

    # Read trajectory -------
    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)
    traj_dl = DataLoader(traj_, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=4)
    print('_'*70+'\n')

    # Train model -----------
    model.epoch_losses = list(np.loadtxt(model.loss_path, usecols=1))
    checkpoint_callback = ModelCheckpoint(
        dirpath=out,
        filename=f'{fname}checkpoint',
        save_top_k=1  # Save the best checkpoint
        )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=n_devices,
        callbacks=[checkpoint_callback]
        )
    
    trainer.fit(model, traj_dl, ckpt_path=checkpoint)

# -------------------------------------------- 
#            D E C O M P R E S S                  
# -------------------------------------------- 

def decompress(top:str, model:str, compressed:str, out:str):
    r'''
decompress compressed-trajectory
--------------------------------
top (str) : Path to the topology file (parm7|pdb)
model (str) : Path to the saved model file
compressed (str) : Path to the compressed trajectory file
out (str) : Output trajectory file path with name. Use extention to define file type (*.nc|*.xtc)
    '''
    pathExists(top)
    pathExists(model)
    pathExists(compressed)
    pathExists(os.path.dirname(out))
        
    if top.endswith('parm7'):
        top = md.load_prmtop(top)
    elif top.endswith('pdb'):
        top = md.load_pdb(top).topology
    else:
        raise ValueError('Supported formats: .parm7, .pdb')
    
    # Define device ---------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Device name:', torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        print('Device name: CPU')
    
    # Decompress ------------
    decoder = torch.load(model).to(device).model.decoder
    compressed = torch.concatenate(pickle.load(open(compressed, 'rb')))
    
    with torch.no_grad():
        decoder.eval()
    
        if out.endswith('.nc'):
            traj_file = md.formats.netcdf.NetCDFTrajectoryFile(out, 'w')
        elif out.endswith('.xtc'):
            traj_file = md.formats.xtc.XTCTrajectoryFile(out, 'w')
        else:
            raise ValueError('Supported formats: .nc, .xtc')
    
        with traj_file as f:
            for i in tqdm(range(len(compressed)), desc='Compressing '):
                np_traj_frame = decoder(comp[i].reshape(1, -1)).detach().cpu().numpy()
                np_traj_frame = np_traj_frame.reshape(-1, np_traj_frame.shape[2], 3)
                f.write(np_traj_frame)
    
    print('\n')