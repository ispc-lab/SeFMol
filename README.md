# SeFMol
The official repository of our paper "Steering Semi-flexible Molecular Diffusion Model for Structure-Based Drug Design with Reinforcement Learning"


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
