# MolZip
> ### Author: Namindu De Silva
___
<Description>

## Dependencies

<div style="display: flex;">
<div style="width: 50%;">
<ul>
<li>python >= 3.6.0</li>
<li>wheel</li>
<li>torch</li>
<li>torchvision</li>
<li>torchaudio</li>
</ul>
</div>
<div style="width: 50%;">
<ul>
<li>pytorch-lightning</li>
<li>scikit-learn</li>
<li>mdtraj</li>
<li>numpy</li>
<li>tqdm</li>
</ul>
</div>
</div>
<!-- - 
- wheel
- mdtraj
- torch
- torchvision
- torchaudio
- pytorch-lightning
- scikit-learn
- numpy
- tqdm -->

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