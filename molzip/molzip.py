import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
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
#                  T R A I N                  
# -------------------------------------------- 

def train(traj:str, top:str, stride:int=1, out:str=os.getcwd(), fname:str='', epochs:int=100, batchSize:int=128, lat:int=20, memmap:bool=False, cluster:dict=None):
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
memmap (bool) : Use memory-map to read trajectory [Default=False]
cluster (dict) : Dictionary of clusters based on 3D coordinates [Default=None] # get best number of clusters using silhouette score
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
        n_devices = 1
        print('CUDA is not available')

    # Read trajectory -------
    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)
    n_atoms = traj_.shape[2]
    
    if cluster!=None:
        traj_dl = [DataLoader(traj_[cluster[c],:,:,:], batch_size=batchSize, shuffle=False, drop_last=True, num_workers=4) for c in range(len(cluster))]
    else:
        cluster = {0:np.arange(traj_.shape[0]).tolist()}
        traj_dl = [DataLoader(traj_, batch_size=batchSize, shuffle=False, drop_last=False, num_workers=4)]
        
    print('_'*70+'\n')

    pickle.dump(cluster, open(out+fname+"clusters.pkl", 'wb'))

    models = {}
    checkpoint_dict = {}
    for c in range(len(traj_dl)):
    # Train model -----------
        model = AE(n_atoms=n_atoms, latent_dim=lat)
        model = LightAE(model=model, lr=1e-4, loss_path=out+fname+f'losses_{c}.dat')

        print(f'Training Deep Convolutional AutoEncoder model {c}')
    
        checkpoint_callback = ModelCheckpoint(
            dirpath=out,
            filename=f'{fname}checkpoint_{c}',
            save_top_k=1  # Save the best checkpoint
            )
        
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=fname+"logs/")
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=n_devices,
            callbacks=[checkpoint_callback],
            logger=tb_logger
            )

        trainer.fit(model, traj_dl[c])
        models[c] = model
        # rmsd, r2, mean_squared_error = fitMetrics(model=model, dl=traj_dl, top=top)

        checkpoint = os.path.join(out, f'{fname}checkpoint_{c}.ckpt')
        checkpoint_dict[c] = checkpoint
        
        print('\n')
    
    # Clean -----------------
        if os.path.exists(fname+"logs/"):
            shutil.rmtree(fname+"logs/")
        if os.path.exists('temp_traj.dat'):
            os.remove('temp_traj.dat')

        print('\n')

    torch.save(models, out+fname+'model.pt')
    torch.save(checkpoint_dict, out+fname+f'checkpoints.pt')
 
# -------------------------------------------- 
#        C O N T I N U E  T R A I N                  
# -------------------------------------------- 

def cont_train(traj:str, top:str,  model:str, checkpoint:str, cluster:str, stride:int=1, epochs:int=100, batchSize:int=128, lat:int=20, w:float=1.0, memmap:bool=False):
    r'''
compressing trajectory
----------------------
traj (str) : Path to the trajectory file
top (str) : Path to the topology file
cluster (str) : Path to pre-saved clusters.pkl file
stride (int) : Read every strid-th frame [Default=1]
model (str)  = Path to previously trained model file
checkpoint (str) = Path to check point file
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

    # Load model
    models = torch.load(model, weights_only=False)
    checkpoint_dict = torch.load(checkpoint)
    clusters = pickle.load(open(cluster, "rb"))

    # Read trajectory -------
    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)

    for c in range(len(models)):
        model_ = models[c].to(device)
        out = os.path.dirname(model_.loss_path)
        fname = os.path.basename(model_.loss_path).split('_')[0] + '_'

        traj_dl = DataLoader(traj_[clusters[c],:,:,:], batch_size=batchSize, shuffle=True, drop_last=False, num_workers=4)
    
        print('_'*70+'\n')

    # Train model -----------
        model_.epoch_losses = list(np.loadtxt(model_.loss_path, usecols=1))
    
        checkpoint_callback = ModelCheckpoint(
            dirpath=out,
            filename=f'{fname}checkpoint{c}',
            save_top_k=1  # Save the best checkpoint
            )
        
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=fname+"logs/")
        
        trainer = pl.Trainer(
            max_epochs=len(model_.epoch_losses)+epochs,
            accelerator=accelerator,
            devices=n_devices,
            callbacks=[checkpoint_callback],
            logger=tb_logger
            )
        
        trainer.fit(model_, traj_dl, ckpt_path=checkpoint_dict[c])
        models[c] = model_

        checkpoint = torch.load(os.path.join(out, f'{fname}checkpoint_{c}.ckpt'))
        checkpoint_dict[c] = checkpoint
    
    # Clean -----------------
        if os.path.exists(fname+"logs/"):
            shutil.rmtree(fname+"logs/")
        if os.path.exists('temp_traj.dat'):
            os.remove('temp_traj.dat')

        print('\n')

    torch.save(models, model)
    torch.save(checkpoint_dict, os.path.join(out, f'{fname}checkpoints.pt'))
    

# -------------------------------------------- 
#             C O M P R E S S                  
# -------------------------------------------- 
def compress(traj:str, top:str, model:str, cluster:str, stride:int=1, out:str=os.getcwd(), fname:str=None, memmap:bool=False):
    r'''
compressing trajectory
----------------------
traj (str) : Path to the trajectory file
top (str) : Path to the topology file
cluster (str) : Path to pre-saved clusters.pkl file
model (str)  = Path to previously trained model file
stride (int) : Read every strid-th frame [Default=1]
out (str) : Path to save compressed files [Default=current directory]
fname (str) : Prefix for all generated files [Default=None]
memmap (bool) : Use memory-map to read trajectory [Default=False]
        '''
    
    pathExists(traj)
    pathExists(top)
    
    # Define device ---------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_devices = torch.cuda.device_count()
        if n_devices == 1:
            print('Device name:', torch.cuda.get_device_name(device))
        else:
            print(f'Available devices: {n_devices:>02d}')
            for i in range(n_devices):
                print(torch.cuda.get_device_name(i))
    else:
        device = torch.device('cpu')
        n_devices = 1
        print('CUDA is not available')
    
    # Load model
    models = torch.load(model, weights_only=False)
    clusters = pickle.load(open(cluster, "rb"))
    
    if fname != None:
        fname += '_'
    else:
        fname = os.path.basename(models[0].loss_path).split('_')[0] + '_'
        
    if platform.system() == "Windows":
        if not out.endswith('\\'):
            out += '\\'
    else:
        if not out.endswith('/'):
            out += '/'
        
    print('_'*70+'\n')
    

    # Read trajectory -------
    traj_ = read_traj(traj_=traj, top_=top, stride=stride, memmap=memmap)
    traj_dl = DataLoader(traj_, batch_size=1, shuffle=False, drop_last=False, num_workers=4) 

    for c in range(len(models)):
        model_ = models[c]
        encoder = model_.model.encoder.to(device)
        models[c] = encoder

    c_index = get_cluster(clusters)
    z = []
    with torch.no_grad():
        for encoder in models.values():
            encoder.eval()
        for batch, c in tqdm(zip(traj_dl, c_index), desc="Compressing "):
            batch = batch.to(device=device, dtype=torch.float32)
            z.append(models[c](batch))
    
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
    

# -------------------------------------------- 
#            D E C O M P R E S S                  
# -------------------------------------------- 

def decompress(top:str, model:str, compressed:str, out:str, cluster:str):
    r'''
decompress compressed-trajectory
--------------------------------
top (str) : Path to the topology file (parm7|pdb)
model (str) : Path to the saved model file
compressed (str) : Path to the compressed trajectory file
out (str) : Output trajectory file path with name. Use extention to define file type (*.nc|*.xtc)
cluster (str) : Path to pre-saved clusters.pkl file
    '''
    pathExists(top)
    pathExists(model)
    pathExists(compressed)
    pathExists(os.path.dirname(out))
        
    # if top.endswith('parm7'):
    #     top = md.load_prmtop(top)
    # elif top.endswith('pdb'):
    #     top = md.load_pdb(top).topology
    # else:
    #     raise ValueError('Supported formats: .parm7, .pdb')
    
    # Define device ---------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Device name:', torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
        print('Device name: CPU')

    # Load model
    models = torch.load(model, weights_only=False)
    clusters = pickle.load(open(cluster, "rb"))
    c_index = get_cluster(clusters)
    
    # Decompress ------------
    for c in range(len(models)):
        model_ = models[c]
        decoder = model_.model.decoder.to(device)
        models[c] = decoder
        
    compressed = torch.concatenate(pickle.load(open(compressed, 'rb')))
    
    with torch.no_grad():
        for decoder in models.values():
            decoder.eval()
    
        if out.endswith('.nc'):
            traj_file = md.formats.netcdf.NetCDFTrajectoryFile(out, 'w')
        elif out.endswith('.xtc'):
            traj_file = md.formats.xtc.XTCTrajectoryFile(out, 'w')
        else:
            raise ValueError('Supported formats: .nc, .xtc')
    
        with traj_file as f:
            for i in tqdm(range(len(compressed)), desc='Decompressing '):
                np_traj_frame = models[c_index[i]](compressed[i].reshape(1, -1)).detach().cpu().numpy()
                np_traj_frame = np_traj_frame.reshape(-1, np_traj_frame.shape[2], 3)
                f.write(np_traj_frame*10) # make angstrom
    
    print('\n')
    print('Decompression complete\n') 
