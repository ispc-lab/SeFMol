import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset
from .pdbbind import PDBBindDataset
from .ligand_dataset import LigandDataset

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pdbbind':
        dataset = PDBBindDataset(root, *args, **kwargs)
    elif name == 'ligand':
        dataset = LigandDataset(root, *args, **kwargs)
        if 'split' in config:
            split_by_name = torch.load(config.split)
            split = {k: [dataset.name2id[n] for n in names if n in dataset.name2id] for k, names in split_by_name.items()}
            subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
            return dataset, subsets   
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset