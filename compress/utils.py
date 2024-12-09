import numpy as np
import mdtraj as md

def read_traj(xtc_:str, top_:str, ref_:str=None):
    traj = md.load(xtc_, top=top_)
    
    # Remove rotation: algn on 1st frame of the trajectory or a given reference pdb frame
    traj.superpose(traj, 0 if ref_ == None else ref_) 
    
    # Remove translation: Center the COM of each frame
    for i in range(traj.n_frames):
        frame_positions = traj.xyz[i, :, :]
        com = np.mean(frame_positions, axis=0)
        traj.xyz[i, :, :] -= com
    
    return traj.xyz, traj.n_atoms

def minMaxScale(array:np.ndarray):
    return (array - array.min()) / (array.max() - array.min())