import warnings
warnings.filterwarnings("ignore")

import torch
import pickle
import numpy
import argparse
import os
import tqdm

from .utils import *
from .autoencoder import *

torch.set_float32_matmul_precision('medium')

# Inputs ----------------
parser = argparse.ArgumentParser(description="Decompress compressed-trjectories")

parser.add_argument('-m', '--model', type=str, required=True, help='Path to the saved model file')
parser.add_argument('-c', '--comp', type=str, required=True, help='Path to the compressed trajectory file')
parser.add_argument('-p', '--top', type=str, required=True, help='Path to the topology file (parm7|pdb)')
parser.add_argument('-o', '--out', type=str, required=True, help='Output trajectory file path with name. Use extention to define file type (*.nc|*.xtc)')

args = parser.parse_args()
top = args.top
comp = args.comp
model = args.model
out = args.out
del args

if top.endswith('parm7'):
    top = md.load_prmtop(top)
elif top.endswith('pdb'):
    top = md.load_pdb(top).topology
else:
    raise ValueError('This program supports only topology formats: .parm7, .pdb')

# Define device ---------
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Device name:', torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print('Device name: CPU')

# Decompress ------------
decoder = torch.load(model).to(device).model.decoder
comp = torch.concatenate(pickle.load(open(comp, 'rb')))

with torch.no_grad():
    decoder.eval()

    if out.endswith('.nc'):
        traj_file = md.formats.netcdf.NetCDFTrajectoryFile(out, 'w')
    elif out.endswith('.xtc'):
        traj_file = md.formats.xtc.XTCTrajectoryFile(out, 'w')
    else:
        raise ValueError('Supported formats: .nc, .xtc')

    with traj_file as f:
        for i in tqdm(range(len(comp)), desc='Compressing'):
            np_traj_frame = decoder(comp[i].reshape(1, -1)).detach().cpu().numpy()
            np_traj_frame = np_traj_frame.reshape(-1, np_traj_frame.shape[2], 3)
            f.write(np_traj_frame)

# End -------------------
print('\n')
