import warnings
warnings.filterwarnings("ignore")

import torch
import pickle
import numpy
import argparse
import os

from utils import *
from autoencoder import *

torch.set_float32_matmul_precision('medium')

# Inputs ----------------
parser = argparse.ArgumentParser(description="Decompress compressed-trjectories")

parser.add_argument('-m', '--model', type=str, required=True, help='Path to the saved model file')
parser.add_argument('-c', '--comp', type=str, required=True, help='Path to the compressed trajectory file')
parser.add_argument('-p', '--top', type=str, required=True, help='Path to the topology file [parm7|pdb]')

args = parser.parse_args()
top = args.top
comp = args.comp
model = args.model
del args

if top.endswith('parm7'):
    top = md.load_prmtop(top)
elif top.endswith('pdb'):
    top = md.load_pdb(top).topology
else:
    raise ValueError('This program supports only topology formats: [ parm7 | pdb ] ')

decoder = torch.load(model).to(device).model.decoder
comp = torch.concatenate(pickle.load(open(comp, 'rb')))

# Define device ---------
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Device name:', torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print('Device name: CPU')

# Decompress ------------
with torch.no_grad():
    decoder.eval()
    np_traj = decoder(comp).detach().cpu().numpy()
    np_traj = np_traj.reshape(-1,np_traj.shape[2],3)

# Save trajectories
md.Trajectory(np_traj, top).save_netcdf('output_trajectory.nc')