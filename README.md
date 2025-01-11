# MolZip
### Author: Namindu De Silva, Alberto Perez

<Description>

## Dependencies
The `dim` library makes extensive use of `numpy` and `sklearn` and `graphtime`.
- python >= 3.6.0
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

## Usage
1. Clone the repository:
```
git clone https://github.com/PDNALab/DIM.git 
```

2. Append path: Befor importing DIM module, please append the path as shown.
```
import sys
sys.path.append('<path to dim>')

from dim import utils
from dim import dimgen as dim 
```

3. Main functions:
- Make dim object for arbitary DNA sequence. [sequence is given 5'-3']
```
# Load DMRF object - pre learned from ABC data
with open('<path to dim>/dim/gen_data/dmrf_tetramer_20_4.dmrf', 'rb') as f:
    dmrf = pickle.load(f)

# Make dim object
DNA = dim.dim(seq='ATGCATGC', dmrf=dmrf)
```
- Free energy
```
# For smaller DNA sequences:
Free_energy1 = DNA.get_free_energy1()

# When DNA sequences have more sub-systems [faster method]:
Free_energy2 = get_free_energy2(cut=10)
```
- Transition matrix
```
T_mat = DNA.get_transition_matrix()
```
install pytorch packages with cuda: https://www.restack.io/p/pytorch-lightning-answer-cuda-11-7-cat-ai
