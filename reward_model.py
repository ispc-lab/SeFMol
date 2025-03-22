
# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from rdkit import DataStructs


def norm_pharm_score(x):
    # clip pharm score  
    x = min(max(x, 0), 500)
    return x / 500

def calculate_reward_org_vina(gen_mol, pdb_path):
  try:
    smiles = Chem.MolToSmiles(gen_mol)
    if '.' in smiles:
      return 0  # invalid mol, reward=-1
    
    vina_task = VinaDockingTask.from_generated_mol(gen_mol, pdb_path)
    score_only_results = vina_task.run(mode='score_only', exhaustiveness=32)[0]
    x = score_only_results['affinity']
    if x > 0:
      x = 0
      
    ret = x
  except:
      ret = 0
  return ret

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


def calculate_reward_drugbank_vina(gen_mol, pdb_path, max_reward=0, min_reward=-20, 
                              ref_fps_path='data/drugbank_fps.pt'):
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
    
        # cal sim with drugbank
        ref_fps_list = torch.load(ref_fps_path)
        input_fps = Chem.RDKFingerprint(gen_mol)
        for ref_fps in ref_fps_list:
            sim_score = DataStructs.TanimotoSimilarity(input_fps, ref_fps)

        # ret = vina_score * 0.25 + qed_score * 0.25 + sa_score * 0.25 + sim_score * 0.25
        ret = vina_score * 0.75 + sim_score * 0.25
        # ret = vina_score
    except:
        ret = 0
    return ret


# def calculate_reward_drugbank_mi_vina(gen_mols_list, pdb_path, ref_repr_path='data/drugbank_unimol_repr.pt', device='cpu'):
#     input_mols = []
#     vina_list, qed_list, sa_list = [], [], []
#     for mol in gen_mols_list:
#         try:
#             smiles = Chem.MolToSmiles(mol)
#             if '.' in smiles:
#                 continue    # invalid mol
 
#             vina_task = VinaDockingTask.from_generated_mol(mol, pdb_path)
#             x = vina_task.run(mode='score_only', exhaustiveness=32)[0]['affinity']
#             # clip vina score
#             if x < -20:
#                 x = -20
#             elif x > 0:
#                 x = 0
#             vina_list.append(- x / 20)
#             qed_list.append(qed(mol))
#             sa_list.append(compute_sa_score(mol))
        
#             input_mols.append(mol)
            
#         except:
#             continue
        
#     # cal mutual information with drugbank
#     from utils.cal_mi import get_mi, mol_to_atom_coords_dict, get_unimol_repre
    
#     ref_repr_list = torch.load(ref_repr_path)
#     gen_results = mol_to_atom_coords_dict(input_mols)
#     gen_unimol_reprs = get_unimol_repre(gen_results, device=device)
     
#     mi_total = get_mi(gen_unimol_reprs, ref_repr_list)
#     mi_list = []

#     for i in range(len(gen_unimol_reprs)):
#         repres_minus_cur_repr = np.concatenate((gen_unimol_reprs[:i], gen_unimol_reprs[i+1:]), axis=0)
#         mi_without_repr = get_mi(repres_minus_cur_repr, ref_repr_list)
#         mi_list.append(mi_total - mi_without_repr)
#     print(mi_total)
#     print(vina_list)
#     print(qed_list)
#     print(sa_list)
#     print(mi_list)
#     reward = (0.7 * torch.tensor(vina_list) 
#               + 0.1 * torch.tensor(qed_list) 
#               + 0.1 * torch.tensor(sa_list) 
#               + 0.1 * torch.tensor(mi_list)
#             )
#     print(reward)
#     return reward

def calculate_reward_drugbank_mi_vina(gen_mols_list, pdb_path, ref_repr_path='data/drugbank_unimol_repr.pt', device='cpu'):
    input_mols = []
    vina_list, qed_list, sa_list, mi_list = [], [], [], []
    
    # 初始化reward列表，所有元素先设为0
    reward_list = [0] * len(gen_mols_list)
    
    for idx, mol in enumerate(gen_mols_list):
        try:
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:  # invalid mol
                continue

            vina_task = VinaDockingTask.from_generated_mol(mol, pdb_path)
            x = vina_task.run(mode='score_only', exhaustiveness=32)[0]['affinity']
            # clip vina score
            x = max(min(x, 0), -20)
            vina_score = -x / 20
            qed_score = qed(mol)
            sa_score = compute_sa_score(mol)
        
            input_mols.append(mol)
            vina_list.append(vina_score)
            qed_list.append(qed_score)
            sa_list.append(sa_score)
            
            # 对于有效的分子，先设置其reward为1，稍后再计算实际值
            reward_list[idx] = 1
            
        except:
            continue
    
    # 计算 mutual information
    from utils.cal_mi import get_mi, mol_to_atom_coords_dict, get_unimol_repre
    
    ref_repr_list = torch.load(ref_repr_path)
    gen_results = mol_to_atom_coords_dict(input_mols)
    gen_unimol_reprs = get_unimol_repre(gen_results, device=device)
    
    try:
        mi_total = get_mi(gen_unimol_reprs, ref_repr_list)
    except:
        mi_total = 0

    for i in range(len(gen_unimol_reprs)):
        try:
            repres_minus_cur_repr = np.concatenate((gen_unimol_reprs[:i], gen_unimol_reprs[i+1:]), axis=0)
            mi_without_repr = get_mi(repres_minus_cur_repr, ref_repr_list)
            if mi_total:
                mi_list.append(mi_total - mi_without_repr)
            else:
                mi_list.append(0)
        except:
            mi_list.append(0)
    
    # alpha = 1.e+13
    # print(mi_total)
    # print(torch.tensor(mi_list))
    # 计算有效分子的reward
    for idx, is_valid in enumerate(reward_list):
        if is_valid == 1:  # 对于有效分子，计算实际reward
            reward_list[idx] = (0.7 * vina_list.pop(0) 
                                + 0.1 * qed_list.pop(0) 
                                + 0.1 * sa_list.pop(0) 
                                + 0.1 * mi_list.pop(0))

    reward = torch.tensor(reward_list)
    # print(reward)
    return reward



def calculate_reward_vendi_vina(gen_mols_list, pdb_path, device='cpu'):
    input_mols = []
    vina_list, qed_list, sa_list, vendi_list = [], [], [], []
    
    # 初始化reward列表，所有元素先设为0
    reward_list = [0] * len(gen_mols_list)
    
    for idx, mol in enumerate(gen_mols_list):
        try:
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:  # invalid mol
                continue
            vina_task = VinaDockingTask.from_generated_mol_reward(mol, pdb_path)
            x = vina_task.run(mode='score_only', exhaustiveness=32)[0]['affinity']
            # clip vina score
            x = max(min(x, 0), -20)
            vina_score = -x / 20
            qed_score = qed(mol)
            sa_score = compute_sa_score(mol)
        
            input_mols.append(mol)
            vina_list.append(vina_score)
            qed_list.append(qed_score)
            sa_list.append(sa_score)
            
            # 对于有效的分子，先设置其reward为1，稍后再计算实际值
            reward_list[idx] = 1
            
        except:
            continue
    
    from utils.unimol_repre import get_vendi_unimol, get_unimol_repre
    gen_unimol_reprs = get_unimol_repre(input_mols, device)
    vendi_total = get_vendi_unimol(gen_unimol_reprs)

    for i in range(len(gen_unimol_reprs)):
        repres_minus_cur_repr = torch.concatenate((gen_unimol_reprs[:i], gen_unimol_reprs[i+1:]), axis=0)
        vendi_without_repr = get_vendi_unimol(repres_minus_cur_repr)
        if vendi_total:
            vendi_list.append(vendi_total - vendi_without_repr)
        else:
            vendi_list.append(0)
   
    
    # 计算有效分子的reward
    print("vina_list", str(torch.tensor(vina_list)))
    print("vendi_list", str(torch.tensor(vendi_list)))
    for idx, is_valid in enumerate(reward_list):
        if is_valid == 1:  # 对于有效分子，计算实际reward
            reward_list[idx] = (0.8 * vina_list.pop(0) 
                                + 0.1 * qed_list.pop(0) 
                                + 0.1 * sa_list.pop(0) 
                                + 10 * vendi_list.pop(0))
    reward = torch.tensor(reward_list).float().to(device)
    return reward


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
