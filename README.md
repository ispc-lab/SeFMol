# SeFMol
:boom: The official repository of our paper "Steering Semi-flexible Molecular Diffusion Model for Structure-Based Drug Design with Reinforcement Learning". 

## Platform
In addition to providing code for training and inference of SeFMol, we also offer a user-friendly visualization interface. Once the paper is accepted, we will release the complete platform.

<p align="center">
  <img width="700" src="figs/platform.png" /> 
</p>

## Prerequisites
We have presented the conda environment file in `./environment.yml`.

## Install via Conda and Pip
```
conda create -n SeFMol python=3.9
conda activate SeFMol
conda install pytorch==1.13.1  pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge pdbfixer
conda install conda-forge::openbabel

pip isntall protobuf==5.27.1
pip install networkx==3.2.1
pip install rdkit==2023.9.6
pip install biopython==1.83

```

## Data
The data used for training / evaluating the model are organized in the [data](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link) Google Drive folder.

To train the model from scratch, you need to download the preprocessed lmdb file and split file:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, you need to download _and_ unzip the `test_set.zip`. It includes the original PDB files that will be used in Vina Docking.

## Training
Ridid pre-training:
```
python train_rigid_pt.py  
```

Ridid finetuing:
```
python train_rigid_ft.py
```

Semi-flexible training:
```
python train_sfrl.py
```


## Sampling
```
python sample.py --config configs/rl.yml --start_index 0  --end_index 99 
```

## Evaluation
### Evaluation from sampling results
```
python eval_split_diff.py
```
