import os
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from easydict import EasyDict
from tqdm import tqdm
from copy import deepcopy
from rdkit.Chem.FilterCatalog import *



def inverse_scale_vina_score(scaled_score, min_score=-15.0, max_score=10.0):
    original_score = min_score + (1 - scaled_score) * (max_score - min_score)
    return original_score


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = Crippen.MolLogP(mol)
    rule_4 = (logp >= -2) and (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
    

def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    for confId in confIds:
        AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
        rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
        rmsd_list.append(rmsd)
    # mol3d = Chem.RemoveHs(mol3d)
    rmsd_list = np.array(rmsd_list)
    return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = Crippen.MolLogP(mol)
    hacc_score = Lipinski.NumHAcceptors(mol)
    hdon_score = Lipinski.NumHDonors(mol)
    return qed_score, sa_score, logp_score, hacc_score, hdon_score


class SimilarityWithTrain:
    def __init__(self, mol, config):
        self.train_smiles = None
        self.train_fingers = None
        self.mol = mol
        self.cfg_dataset = config.dataset
       
    def _get_train_mols(self):
        file_not_exists = (not os.path.exists(self.cfg_dataset.train_fingerprint)) or (not os.path.exists(self.cfg_dataset.train_smiles))
        if file_not_exists:
            _, subsets = get_dataset(config = self.cfg_dataset)
            train_set = subsets['train']
            self.train_smiles = []
            self.train_fingers = []
            for data in tqdm(train_set):  # calculate fingerprint and smiles of train data
                mol_path = self.cfg_dataset['path'] + data['ligand_filename']
                mol = Chem.MolFromMolFile(mol_path)  # automately sanitize 
                smiles = Chem.MolToSmiles(mol)
                fg = Chem.RDKFingerprint(mol)
                self.train_fingers.append(fg)
                self.train_smiles.append(smiles)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)
            torch.save(self.train_smiles, self.cfg_dataset.train_smiles)
            torch.save(self.train_fingers, self.cfg_dataset.train_fingerprint)
        else:
            self.train_smiles = torch.load(self.cfg_dataset.train_smiles)
            self.train_fingers = torch.load(self.cfg_dataset.train_fingerprint)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)

    def _get_uni_mols(self):
        self.train_uni_smiles, self.index_in_train = np.unique(self.train_smiles, return_index=True)
        self.train_uni_fingers = [self.train_fingers[idx] for idx in self.index_in_train]

    def get_similarity(self, mol):
        if self.train_fingers is None:
            self._get_train_mols()
            self._get_uni_mols()
        # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize 
        fp_mol = Chem.RDKFingerprint(mol)
        sims = [DataStructs.TanimotoSimilarity(fp, fp_mol) for fp in self.train_uni_fingers]
        return np.array(sims)
    
    def get_novelty(self):
        if self.train_fingers is None:
            self._get_train_mols()
            self._get_uni_mols()
        gen_smiles = Chem.MolToSmiles(self.mol) 
        gen_smiles_set = set(gen_smiles) - {None}
        train_set = set(self.train_uni_smiles)
        novelty = len(gen_smiles_set - train_set) / len(gen_smiles_set)
        return novelty

    def get_top_sims(self, top=100):
        similarities = self.get_similarity(self.mol)
        idx_sort = np.argsort(similarities)[::-1]
        top_scores = similarities[idx_sort[:top]]
        # top_smiles = self.train_uni_smiles[idx_sort[:top]]
        # return top_scores, top_smiles
        return top_scores

        
class SimilarityWithPreTrain:
    def __init__(self, mol, config):
        self.train_smiles = None
        self.train_fingers = None
        self.mol = mol
        self.cfg_dataset = config.pretrain_dataset
       
    def _get_train_mols(self):
        file_not_exists = (not os.path.exists(self.cfg_dataset.train_fingerprint)) or (not os.path.exists(self.cfg_dataset.train_smiles))
        if file_not_exists:
            _, subsets = get_dataset(config = self.cfg_dataset)
            train_set = subsets['train']
            self.train_smiles = []
            self.train_fingers = []
            for data in tqdm(train_set):  # calculate fingerprint and smiles of train data
                mol_path = self.cfg_dataset['path'] + data['ligand_filename']
                mol = Chem.MolFromMolFile(mol_path)  # automately sanitize 
                smiles = Chem.MolToSmiles(mol)
                fg = Chem.RDKFingerprint(mol)
                self.train_fingers.append(fg)
                self.train_smiles.append(smiles)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)
            torch.save(self.train_smiles, self.cfg_dataset.train_smiles)
            torch.save(self.train_fingers, self.cfg_dataset.train_fingerprint)
        else:
            self.train_smiles = torch.load(self.cfg_dataset.train_smiles)
            self.train_fingers = torch.load(self.cfg_dataset.train_fingerprint)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)

    def _get_uni_mols(self):
        self.train_uni_smiles, self.index_in_train = np.unique(self.train_smiles, return_index=True)
        self.train_uni_fingers = [self.train_fingers[idx] for idx in self.index_in_train]

    def get_similarity(self, mol):
        if self.train_fingers is None:
            self._get_train_mols()
            self._get_uni_mols()
        # mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize 
        fp_mol = Chem.RDKFingerprint(mol)
        sims = [DataStructs.TanimotoSimilarity(fp, fp_mol) for fp in self.train_uni_fingers]
        return np.array(sims)
    
    def get_novelty(self):
        if self.train_fingers is None:
            self._get_train_mols()
            self._get_uni_mols()
        gen_smiles = Chem.MolToSmiles(self.mol) 
        gen_smiles_set = set(gen_smiles) - {None}
        train_set = set(self.train_uni_smiles)
        novelty = len(gen_smiles_set - train_set) / len(gen_smiles_set)
        return novelty

    def get_top_sims(self, top=100):
        similarities = self.get_similarity(self.mol)
        idx_sort = np.argsort(similarities)[::-1]
        top_scores = similarities[idx_sort[:top]]
        # top_smiles = self.train_uni_smiles[idx_sort[:top]]
        # return top_scores, top_smiles
        return top_scores

def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)

def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)


def lipinski(mol):
    count = 0
    if qed(mol) <= 5:
        count += 1
    if Chem.Lipinski.NumHDonors(mol) <= 5:
        count += 1
    if Chem.Lipinski.NumHAcceptors(mol) <= 10:
        count += 1
    if Chem.Descriptors.ExactMolWt(mol) <= 500:
        count += 1
    if Chem.Lipinski.NumRotatableBonds(mol) <= 5:
        count += 1
    return count


def internal_diversity(gen_mol, mols, device='cpu'):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    fgs = []

    for mol in mols:
        fgs.append(Chem.RDKFingerprint(mol))
    fgs = np.array(fgs)

    gen_mol = np.array([Chem.RDKFingerprint(gen_mol)])
    return 1 - (average_agg_tanimoto(gen_mol, fgs, agg='mean', device=device)).mean()
