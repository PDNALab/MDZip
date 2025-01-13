# MolZip
___
#### _Author: Namindu De Silva_
[Add Description Here]

## Dependencies

- wheel
- mdtraj
- torch
- torchvision
- torchaudio
- pytorch-lightning
- scikit-learn
- numpy
- tqdm

## Installation
### Linux/Windows with CUDA
Create conda environment
```
conda create -n <my-env> python=3.10
conda activate <my-env>
```
Install dependencies (recomended for CUDA build)
```
conda install -c confa-forge mdtraj
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install lightning
```
Install MolZip
```
git clone https://github.com/nami-rangana/MolZip.git
cd MolZip
python setup.py sdist bdist_wheel
cd dist
pip insall molzip-0.1.0-py3-none-any.whl
```
### Linux/Windows/OSX without CUDA
```
conda create -n <my-env> python=3.10
conda activate <my-env>
conda install -c confa-forge mdtraj

git clone https://github.com/nami-rangana/MolZip.git
cd MolZip
python setup.py sdist bdist_wheel
cd dist
pip insall molzip-0.1.0-py3-none-any.whl
```

## Help
```
molzip.compress()

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
```
```
molzip.decompress()

decompress compressed-trajectory
--------------------------------
top (str) : Path to the topology file (parm7|pdb)
model (str) : Path to the saved model file
compressed (str) : Path to the compressed trajectory file
out (str) : Output trajectory file path with name. Use extention to define file type (*.nc|*.xtc)
```

## Cite

Fill here