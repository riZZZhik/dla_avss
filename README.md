# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains several architectures and training configs for the task of **speech separation** in the time domain:

- **DPRNN** — dual-path recurrent network for long-context modeling;
- **Conv-TasNet–like model** — convolutional TCN-based separator in the time domain;
- **AVRTFSNet (audio-visual RTFS)** — recurrent time–frequency model that additionally uses visual embeddings (mouth crops).

Best checkpoints can be found in a DEMO.ipynb 


## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

The code is designed to work with the DLA AVSS dataset in the following directory structure:
dla_dataset
├── audio
│   ├── mix
│   │   ├── <id>.wav
│   │   └── ...
│   ├── s1
│   │   ├── <id>.wav
│   │   └── ...
│   └── s2
│       ├── <id>.wav
│       └── ...
└── mouths
    ├── <speaker_id>.npz      # mouth crops or embeddings (for AV models)
    └── ...


In Hydra configs this path is typically referenced as:
  ```
datasets:
  train:
    data_dir: "PATH_TO/dla_dataset"
  val:
    data_dir: "PATH_TO/dla_dataset"
  ```
## How To Use
All training and evaluation scripts are built on top of Hydra.
Main entry points:
train.py — training loop (with validation and checkpointing)
inference.py — running inference / computing metrics for saved predictions 

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```
DPRNN Baseline:
```
python3 train.py -cn=baseline_dprnn \
  datasets.train.data_dir=/path/to/dla_dataset \
  datasets.val.data_dir=/path/to/dla_dataset \
  dataloader.batch_size=8 \
  trainer.n_epochs=100
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
