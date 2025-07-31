from setuptools import setup, find_packages

setup(
    name='MDZip',
    version='0.1.0',
    author='Namindu De Silva',
    author_email='nami.rangana@gmail.com',
    description='Compress MD trajectories using deep convolutional autoencoder',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/nami-rangana/MolZip.git',
    packages=find_packages(),
    classifiers=[               # Optional classifiers for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[  
        'wheel',
        'mdtraj',
        'mdanalysis',
        'torch',
        'torchvision',
        'torchaudio',
        'pytorch-lightning',
        'scikit-learn',
        'numpy',
        'tqdm'
    ]
)