"""Reward model."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rdkit.Chem.Descriptors import qed
from utils.sascorer import compute_sa_score
from models.common import compose_context
from models.sefmol_sfrl import decoup_batch, unpadding_func
from utils.evaluation.docking_vina import VinaDockingTask
from rdkit import Chem
from models.value_tf import TransformerEncoder


def calculate_reward_vina(gen_mol, pdb_path):
    try:
        smiles = Chem.MolToSmiles(gen_mol)
        if '.' in smiles:
            return 0, 0, 0  # invalid mol, reward=-1

        vina_task = VinaDockingTask.from_generated_mol_reward(gen_mol, pdb_path)
        x = vina_task.run(mode='score_only', exhaustiveness=32)[0]['affinity']

        # clip vina score  
        x = max(min(x, 0), -20)
        vina_score = -x / 20
        
        qed_score = qed(gen_mol)
        sa_score = compute_sa_score(gen_mol)
        return vina_score, qed_score, sa_score
    except:
        return 0, 0, 0 

def calculate_reward_scale_vina(gen_mol, pdb_path):
    try:
        smiles = Chem.MolToSmiles(gen_mol)
        if '.' in smiles:
            return 0  # invalid mol, reward=-1

        vina_task = VinaDockingTask.from_generated_mol_reward(gen_mol, pdb_path)
        x = vina_task.run(mode='score_only', exhaustiveness=32)[0]['affinity']

        # clip vina score
        x = max(min(x, 0), -20)
        vina_score = - x / 20
  
        qed_score = qed(gen_mol)
        sa_score = compute_sa_score(gen_mol)
        ret = 0.8 * vina_score + 0.1 * qed_score + 0.1 * sa_score
        ret = vina_score
    except:
        ret = 0
    return ret

def calculate_reward_multiple_vina(gen_mol, pdb_path, max_reward=0, min_reward=-20):
    try:
        smiles = Chem.MolToSmiles(gen_mol)
        if '.' in smiles:
            return 0  # invalid mol, reward=-1

        vina_task = VinaDockingTask.from_generated_mol(gen_mol, pdb_path)
        x = vina_task.run(mode='score_only', exhaustiveness=32)[0]['affinity']

        # clip vina score
        if x < min_reward:
            x = min_reward
        elif x > max_reward:
            x = max_reward

        vina_score = - x / 20

        qed_score = qed(gen_mol)
        sa_score = compute_sa_score(gen_mol)
        ret = vina_score * qed_score * sa_score
    except:
        ret = 0
    return ret


def pad_tensor(input_tensor, max_node_numbers, hidden_dim):
    batch_size, current_node_numbers, _ = input_tensor.shape
    if current_node_numbers < max_node_numbers:
        padding_size = max_node_numbers - current_node_numbers
        padding = torch.zeros(batch_size, padding_size, hidden_dim, dtype=input_tensor.dtype, device=input_tensor.device)
        padded_tensor = torch.cat([input_tensor, padding], dim=1)
    elif current_node_numbers > max_node_numbers:
        padded_tensor = input_tensor[:, :max_node_numbers, :]
    else:
        padded_tensor = input_tensor
    return padded_tensor
  
  
class ValueMulti(nn.Module):
  """ValueMulti."""

  def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
      super(ValueMulti, self).__init__()
      self.config = config

      # model definition
      self.hidden_dim = config.hidden_dim
      self.num_classes = ligand_atom_feature_dim
      self.max_node_numbers = 700
      # atom embedding
      self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)
      self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']
      self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_dim)

      self.encoder = TransformerEncoder(
            hidden_channels = 128,
            edge_channels = 128,
            key_channels = 128,
            num_heads = 4,
            num_interactions = 6,
            k = 48,
            cutoff = 10.0,
        )

      self.fc1 = nn.Linear(89600, 128)  
      self.fc2 = nn.Linear(128, 64) 
      self.fc3 = nn.Linear(64, 1)  
      
  def forward(self, state, protein_pos_v):
      batch_szie = state.shape[0]
      state, ligand_num_atoms = unpadding_func(state, return_atom_nums=True)

      protein_pos_v, protein_num_atoms = unpadding_func(protein_pos_v, return_atom_nums=True)
      batch_ligand = torch.repeat_interleave(torch.arange(batch_szie), torch.tensor(ligand_num_atoms)).to(state.device)
      batch_protein = torch.repeat_interleave(torch.arange(batch_szie), torch.tensor(protein_num_atoms)).to(state.device)
      
      ligand_pos = state[:, :3]
      ligand_v = torch.squeeze(state[:, 3:], dim=1).long()
      init_ligand_v = F.one_hot(ligand_v, self.num_classes).float()
  
      protein_pos = protein_pos_v[:, :3]
      protein_v = protein_pos_v[:, 3:]

      h_protein = self.protein_atom_emb(protein_v)
      init_ligand_h = self.ligand_atom_emb(init_ligand_v)
      
      h_all, pos_all, batch_all, mask_ligand = compose_context(
          h_protein=h_protein,
          h_ligand=init_ligand_h,
          pos_protein=protein_pos,
          pos_ligand=ligand_pos,
          batch_protein=batch_protein,
          batch_ligand=batch_ligand,
      )
     
      outputs = self.encoder(h_all, pos_all, batch_all)
      outputs = decoup_batch(outputs, batch_all)
      outputs = pad_tensor(outputs, self.max_node_numbers, self.hidden_dim)
      x = outputs.view(outputs.size(0), -1) 
      x = torch.relu(self.fc1(x))  
      x = torch.relu(self.fc2(x))  
      x = self.fc3(x)            

      return x
