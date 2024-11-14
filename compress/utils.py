import numpy as np
import mdtraj as md

def read_traj(xtc_:str, top_:str, ref_:str=None):
    traj = md.load(xtc_, top=top_)
    traj.superpose(traj, 0 if ref_ == None else ref_) # algn on 1st frame of the trajectory or a given reference pdb frame
    for i in range(traj.n_frames):
        # Get the positions of the protein atoms in the current frame
        frame_positions = traj.xyz[i, :, :]
        
        # Compute the center of mass (COM) of the protein in the current frame
        com = np.mean(frame_positions, axis=0)
        
        # Subtract the COM from the positions to center the protein at the origin
        traj.xyz[i, :, :] -= com
    return traj.xyz

def minMaxScale(array:np.ndarray):
    return (array - array.min()) / (array.max() - array.min())