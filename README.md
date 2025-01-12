# MolZip
> ### Author: Namindu De Silva
___
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

## Cite

Fill here