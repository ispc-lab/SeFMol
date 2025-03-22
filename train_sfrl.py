import argparse
import os
import shutil
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader
import utils.train as utils_train
import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.sefmol_sfrl import ScorePosNet3D, log_sample_categorical, unpadding_func, initialize_diffusion_params
from utils.evaluation import atom_num
from diffusers.optimization import get_scheduler
from reward_model import *
from utils import misc, reconstruct, transforms
import dataclasses
import random
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default='test', type=str)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--config', default='configs/rl.yml', type=str)
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--load_ckpt_path', default='./checkpoints/rigid_model.pt', type=str)

parser.add_argument('--save_ckpt_path', default='./checkpoints/test', type=str)
parser.add_argument('--sample_steps', type=int, default=50, help='sample steps of diffusion models')
parser.add_argument('--condition_properties', type=list, default=[1.0, 1.0, 1.0, 50, 3, 2, 0.5, 2])

# hyper paramter
parser.add_argument('--max_train_steps', type=int, default=50000, help='[100000, 50000]Total number of training steps to perform.')
parser.add_argument('--buffer_size', type=int, default=1000)
parser.add_argument('--reward_mode', type=str, default='scale_vina',
                                    choices=['vina', 'scale_vina', 'multiple_vina'])
parser.add_argument('--reward_weight', type=float, default=10, help='pharm 0.1, vina 10')
parser.add_argument('--kl_weight', type=float, default=1.e-2)
parser.add_argument('--kl_warmup', type=int, default=-1, help='warm up for kl weight')
parser.add_argument('--ratio_clip', type=float, default=1.e-2)    
parser.add_argument('--max_ligand_atom_nums', type=int, default=50, help='padding dim')
parser.add_argument('--max_protein_atom_nums', type=int, default=612, help='padding dim')

# policy func
parser.add_argument('--gen_batch_size', type=int, default=10, help='Number of generating samples per perform.')
parser.add_argument('--p_batch_size', type=int, default=8)
parser.add_argument('--p_step', type=int, default=5, help='The number of steps to update the policy per sampling step')
parser.add_argument('--learning_rate', type=float, default=1.e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=-1)
parser.add_argument("--lr_scheduler", type=str, default="constant",
                    help=(
                        'The scheduler type to use. Choose between ["linear", "cosine",'
                        ' "cosine_with_restarts", "polynomial", "constant",'
                        ' "constant_with_warmup"]'))   
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)
parser.add_argument('--adam_weight_decay', type=float, default=0.0)
parser.add_argument('--adam_epsilon', type=float, default=1.e-08)
parser.add_argument('--clip_norm', type=float, default=0.1, help='norm for gradient clipping')
parser.add_argument('--g_step', type=int, default=1, help='The number of sampling steps')

# value func
parser.add_argument('--v_flag', type=int, default=1, help='using of value function')
parser.add_argument('--v_lr', type=float, default=1.e-4)
parser.add_argument('--v_batch_size', type=int, default=16,
                    help='batch size for value function update per gpu, no gradient accumulation')
parser.add_argument('--v_step', type=int, default=5,
                    help='The number of steps to update the value function per sampling step')
                        
args = parser.parse_args()
    

wandb.init(project="SeFMol-project", name=args.project_name)

def _train_policy_func(
    config,
    state_dict,
    moldiff,
    moldiff_copy,
    count,
    policy_steps,
    tpfdata,
    value_function
):
    """Trains the policy function."""
    with torch.no_grad():
        indices = get_random_indices(
            state_dict["state"].shape[0], args.p_batch_size
        )

        batch_state = state_dict["state"][indices]
        batch_next_state = state_dict["next_state"][indices]
        batch_timestep = state_dict["timestep"][indices]
        batch_final_reward = state_dict["final_reward"][indices]
        batch_log_prob = state_dict["log_prob"][indices]
        batch_protein_pos_v = state_dict["protein_pos_v"][indices]

    _, ligand_num_atoms = unpadding_func(batch_state, return_atom_nums=True)
    batch_ligand = torch.repeat_interleave(torch.arange(args.p_batch_size), torch.tensor(ligand_num_atoms)).to(args.device)
    _, protein_num_atoms = unpadding_func(batch_protein_pos_v, return_atom_nums=True)
    batch_protein = torch.repeat_interleave(torch.arange(args.p_batch_size), torch.tensor(protein_num_atoms)).to(args.device)
 
    # calculate loss from the custom function
    pred_log_prob, pred_pos_noise = sample_diffusion_ligand(
            moldiff, args.p_batch_size,
            batch_protein=batch_protein,
            batch_protein_pos_v=batch_protein_pos_v,
            batch_ligand=batch_ligand,
            center_pos_mode=config.model.center_pos_mode,
            log_prob_mode='step_forward_logprob',
            time_steps=batch_timestep,
            state=batch_state,
            next_state=batch_next_state,
            device=args.device)

    _, old_pred_pos_noise = sample_diffusion_ligand(
            moldiff_copy, args.p_batch_size,
            batch_protein=batch_protein,
            batch_protein_pos_v=batch_protein_pos_v,
            batch_ligand=batch_ligand,
            center_pos_mode=config.model.center_pos_mode,
            log_prob_mode='step_forward_logprob',
            time_steps=batch_timestep,
            state=batch_state,
            next_state=batch_next_state,
            device=args.device)

    kl_regularizer_pos = ((pred_pos_noise - old_pred_pos_noise) ** 2).mean(-1)
  
    ratio = torch.exp(pred_log_prob - batch_log_prob)
    ratio = torch.clamp(ratio, 1.0 - args.ratio_clip, 1.0 + args.ratio_clip)
  
    with torch.no_grad():
        if args.v_flag == 1:
            # pylint: disable=line-too-long
            org_reward = batch_final_reward.to(args.device).reshape([args.p_batch_size, 1])
            value_reward = value_function(
                batch_state.float().to(args.device),
                batch_protein_pos_v.to(args.device))
            adv = org_reward - value_reward
        else:
            adv = batch_final_reward.to(args.device).reshape([args.p_batch_size, 1])

    loss = (
        - args.reward_weight
        * adv.detach().float()
        * ratio.float().reshape([args.p_batch_size, 1])
    ).mean()

    if count > args.kl_warmup:
        loss += args.kl_weight * kl_regularizer_pos.mean()
          
    loss.backward()
    
    # logging
    tpfdata.tot_ratio += ratio.mean().item() / policy_steps
    tpfdata.tot_kl += kl_regularizer_pos.mean().item() / policy_steps
    tpfdata.tot_p_loss += loss.item() / policy_steps

@dataclasses.dataclass(frozen=False)
class TrainPolicyFuncData:
  tot_p_loss: float = 0
  tot_ratio: float = 0
  tot_kl: float = 0
  tot_grad_norm: float = 0
  
  
def get_random_indices(num_indices, sample_size):
  """Returns a random sample of indices from a larger list of indices.

  Args:
      num_indices (int): The total number of indices to choose from.
      sample_size (int): The number of indices to choose.

  Returns:
      A numpy array of `sample_size` randomly chosen indices.
  """
  return np.random.choice(num_indices, size=sample_size, replace=False)


def _train_value_func(value_function, state_dict, config):
    """Trains the value function."""
    indices = get_random_indices(state_dict["state"].shape[0], args.v_batch_size)
    # permutation = torch.randperm(state_dict['state'].shape[0])
    # indices = permutation[:v_batch_size]
    batch_state = state_dict["state"][indices]
    batch_timestep = state_dict["timestep"][indices]
    batch_final_reward = state_dict["final_reward"][indices]
    batch_protein_pos_v = state_dict["protein_pos_v"][indices]

    pred_value = value_function(
        batch_state.float().detach().to(args.device),
        batch_protein_pos_v.detach().to(args.device)
    )
    
    batch_final_reward = batch_final_reward.float().to(args.device)
    value_loss = F.mse_loss(
        pred_value.float().reshape([args.v_batch_size, 1]),
        batch_final_reward.detach().reshape([args.v_batch_size, 1]).to(args.device))
    (value_loss / args.v_step).backward()

    del pred_value
    del batch_state
    del batch_timestep
    del batch_final_reward
    del batch_protein_pos_v
    return (value_loss.item() / args.v_step)

def _trim_buffer(buffer_size, state_dict):
    """Delete old samples from the bufffer."""
    if state_dict["state"].shape[0] > buffer_size:
        state_dict["state"] = state_dict["state"][-buffer_size:]
        state_dict["next_state"] = state_dict["next_state"][-buffer_size:]
        state_dict["timestep"] = state_dict["timestep"][-buffer_size:]
        state_dict["final_reward"] = state_dict["final_reward"][-buffer_size:]
        state_dict["log_prob"] = state_dict["log_prob"][-buffer_size:]
        state_dict["protein_pos_v"] = state_dict["protein_pos_v"][-buffer_size:]

def _collect_rollout(config, moldiff, batch, state_dict):
    # Collects trajectories.
  for _ in range(args.g_step):
    # collect the rollout data from the custom sampling function 
        with torch.no_grad():
            mols, pred_pos_traj_list, pred_v_traj_list, log_prob_list = sample_diffusion_ligand(
                moldiff, args.gen_batch_size, batch, device=args.device,
                center_pos_mode=config.model.center_pos_mode,
                log_prob_mode='step_logprob')

            latents_list, protein_pos_v_list = [], []
            
            batch_protein = batch.protein_element_batch
            offset = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            protein_pos = batch.protein_pos - offset[batch_protein]
            protein_v = batch.protein_atom_feature.float()
    
            for i in range(len(mols)):
                current_ligand_atom_nums = pred_pos_traj_list[i].shape[1]
                current_protein_atom_nums = protein_pos.shape[0]
                
                pred_pos_traj_list[i] = F.pad(torch.tensor(pred_pos_traj_list[i]), (0, 0, 0, args.max_ligand_atom_nums - current_ligand_atom_nums))
                pred_v_traj_list[i] = F.pad(torch.tensor(pred_v_traj_list[i]), (0, args.max_ligand_atom_nums - current_ligand_atom_nums))
                latents_list.append(torch.cat([pred_pos_traj_list[i], pred_v_traj_list[i].unsqueeze(2)], dim=2))
                protein_pos_v = F.pad(torch.cat([protein_pos, protein_v], dim=1), (0, 0, 0, args.max_protein_atom_nums - current_protein_atom_nums))
                protein_pos_v_list.append(protein_pos_v)

            latents_list = torch.stack(latents_list).to(args.device)
            protein_pos_v_list = torch.stack(protein_pos_v_list).to(args.device)
            log_prob_list = torch.stack(log_prob_list).to(args.device)
            
            # cal reward 
            pdb_path = config.data.path + batch.protein_filename[0]
            if args.reward_mode == 'vina':
                reward = torch.tensor([calculate_reward_vina(m, pdb_path) for m in mols]).to(args.device)
            elif args.reward_mode == 'scale_vina':
                reward = torch.tensor([calculate_reward_scale_vina(m, pdb_path) for m in mols]).to(args.device)
            elif args.reward_mode == 'multiple_vina':
                reward = torch.tensor([calculate_reward_multiple_vina(m, pdb_path) for m in mols]).to(args.device)
        
            # print('reward', str(reward))
         
            # store the rollout data
            for i in range(latents_list.size(1) - 1):
                # deal with a batch of data in each step i
                state_dict["state"] = torch.cat([state_dict["state"], latents_list[:, i, :, :]], dim=0)     
                state_dict["next_state"] = torch.cat([state_dict["next_state"], latents_list[:, i+1, :, :]], dim=0)
                state_dict["timestep"] = torch.cat((state_dict["timestep"], torch.LongTensor([i]).to(args.device).repeat(latents_list.size(0))))
                state_dict["final_reward"] = torch.cat((state_dict["final_reward"], reward))
                state_dict["log_prob"] = torch.cat((state_dict["log_prob"], log_prob_list[i]))
                state_dict["protein_pos_v"] = torch.cat((state_dict["protein_pos_v"], protein_pos_v_list))

            del (
                mols,
                pred_pos_traj_list,
                pred_v_traj_list,
                latents_list,
                reward,
                log_prob_list,
                offset,
                protein_pos_v_list,
                pdb_path,
                batch_protein,
                protein_pos,
                protein_v
            )
            torch.cuda.empty_cache()



def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

def sample_diffusion_ligand(model, gen_batch_size, data=None, batch_protein=None, batch_protein_pos_v=None, batch_ligand=None, device='cuda:0',
                            num_steps=None, center_pos_mode='protein',
                            log_prob_mode='step_logprob', time_steps=None, state=None, next_state=None):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    n_data = gen_batch_size
    
    if log_prob_mode == 'step_logprob':  
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        batch_protein = batch.protein_element_batch
        protein_pos = batch.protein_pos
        protein_v = batch.protein_atom_feature.float()
        
        pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
        ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
        batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
    else:
        batch_protein = batch_protein
        batch_ligand = batch_ligand
        protein_pos = unpadding_func(batch_protein_pos_v[:, :, :3]).to(args.device)
        protein_v = unpadding_func(batch_protein_pos_v[:, :, 3:]).to(args.device)
    
    # init ligand pos
    center_pos = scatter_mean(protein_pos, batch_protein, dim=0)
    batch_center_pos = center_pos[batch_ligand]
    init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

    # init ligand v
    uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
    init_ligand_v = log_sample_categorical(uniform_logits)
    
    batch_size = batch_ligand.max().item() + 1
    properties = torch.tensor(args.condition_properties).repeat(batch_size).to(device)
    r = model.sample_diffusion(
        protein_pos=protein_pos,
        protein_v=protein_v,
        batch_protein=batch_protein,

        init_ligand_pos=init_ligand_pos,
        init_ligand_v=init_ligand_v,
        batch_ligand=batch_ligand,
        properties=properties,
        time_steps=time_steps,
        state=state,
        next_state=next_state,
        num_steps=num_steps,
        center_pos_mode=center_pos_mode,
        log_prob_mode=log_prob_mode
    )
    
    if log_prob_mode == 'step_forward_logprob':
        return r['log_prob'], r['pred_pos_noise']

    ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']

    # unbatch pos
    ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
    ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
    all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                        range(n_data)]  # num_samples * [num_atoms_i, 3]

    all_step_pos = [[] for _ in range(n_data)]
    for p in ligand_pos_traj:  # step_i
        p_array = p.cpu().numpy().astype(np.float64)
        for k in range(n_data):
            all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_pos = [np.stack(step_pos) for step_pos in
                    all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
    all_pred_pos_traj += [p for p in all_step_pos]

    # unbatch v
    ligand_v_array = ligand_v.cpu().numpy()
    all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

    all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
    all_pred_v_traj += [v for v in all_step_v]
    
    # reconstruction mol
    mol_list, all_pred_pos_traj_list, all_pred_v_traj_list = [], [], []
    for i in range(len(all_pred_pos)):  
        try:
            pred_pos, pred_v = all_pred_pos[i], all_pred_v[i]
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode='add_aromatic')
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
          
            mol_list.append(mol)   
            all_pred_pos_traj_list.append(all_pred_pos_traj[i])
            all_pred_v_traj_list.append(all_pred_v_traj[i])
        except:
            mol_list.append(None)   
            all_pred_pos_traj_list.append(all_pred_pos_traj[i])
            all_pred_v_traj_list.append(all_pred_v_traj[i])
        
    return mol_list, all_pred_pos_traj_list, all_pred_v_traj_list, r['log_prob_traj']

if __name__ == '__main__':
    # Load config
    config = misc.load_config(args.config)  
    misc.seed_all(args.seed)
    
    # Load checkpoint
    ckpt = torch.load(args.load_ckpt_path, map_location=args.device)
    ckpt_ft_path = os.path.join(os.path.dirname(args.save_ckpt_path), 'codes')
    os.makedirs(ckpt_ft_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(ckpt_ft_path, os.path.basename(args.config)))
    shutil.copyfile('train_rl_condition.py', os.path.join(ckpt_ft_path, 'train_rl_condition.py'))
    shutil.copyfile('reward_model.py', os.path.join(ckpt_ft_path, 'reward_model.py'))
    shutil.copyfile('models/molopt_score_model_condition.py', os.path.join(ckpt_ft_path, 'molopt_score_model_condition.py'))
    
    logger = misc.get_logger('RL training', args.save_ckpt_path)
    logger.info(f"Training Config: {config}")
    logger.info(f"Training Args: {args}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode, properties=args.condition_properties)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )

    train_set = [subsets['train'][i] for i in random.sample(range(99990), 10000)]
 
    # load data
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
     
    logger.info(f'Successfully load the dataset (size: {len(train_set)})!')


    # Load moldiff
    initialize_diffusion_params(ckpt['model'], config.model, args.sample_steps)
    moldiff = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        n_timesteps=args.sample_steps,
    ).to(args.device)
    moldiff.load_state_dict(ckpt['model'])
    moldiff.requires_grad_(False)
    moldiff.refine_net.requires_grad_(True)


    # freeze model copy
    moldiff_copy = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        n_timesteps=args.sample_steps,
    ).to(args.device)
    moldiff_copy.load_state_dict(ckpt['model'])
    moldiff_copy.requires_grad_(False)
   
    optimizer = torch.optim.AdamW(
    moldiff.refine_net.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon)
 
    lr_scheduler = get_scheduler(
      args.lr_scheduler,
      optimizer=optimizer,
      num_warmup_steps=args.lr_warmup_steps,
      num_training_steps=args.max_train_steps)

    value_function = ValueMulti(config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim).to(args.device)
    value_optimizer = torch.optim.AdamW(value_function.parameters(), lr=args.v_lr)

    weight_dtype = torch.float32
            
    state_dict = {}
    state_dict["prompt"] = []
    state_dict["state"] = torch.FloatTensor().to(weight_dtype).to(args.device)
    state_dict["next_state"] = torch.FloatTensor().to(weight_dtype).to(args.device)
    state_dict["timestep"] =torch.LongTensor().to(args.device)
    state_dict["final_reward"] = torch.FloatTensor().to(weight_dtype).to(args.device)
    state_dict["log_prob"] = torch.FloatTensor().to(weight_dtype).to(args.device)
    state_dict["protein_pos_v"] = torch.FloatTensor().to(weight_dtype).to(args.device)
    
    count = 0
    buffer_size = args.buffer_size
    policy_steps = args.p_step
  

 
    best_reward = 0      
    for count in range(0, args.max_train_steps // args.p_step):
        batch = next(train_iterator).to(args.device)
        # Sampling manner (releated to sampled ligand atom numbers) 
        _collect_rollout(config, moldiff, batch, state_dict)
        _trim_buffer(buffer_size, state_dict)
        if args.v_flag == 1:
            tot_val_loss = 0
            value_optimizer.zero_grad()
            for v_step in range(args.v_step):
                tot_val_loss += _train_value_func(value_function, state_dict, config)
                value_optimizer.step()
                value_optimizer.zero_grad()
                wandb.log({"value_loss": tot_val_loss}, step=count)
            del tot_val_loss
            torch.cuda.empty_cache()

        # policy learning
        tpfdata = TrainPolicyFuncData()
        for _ in range(args.p_step):
            optimizer.zero_grad()
            _train_policy_func(config, state_dict, moldiff, moldiff_copy, count, policy_steps, tpfdata, value_function)
            norm = torch.nn.utils.clip_grad_norm_(moldiff.refine_net.parameters(), args.clip_norm)
 
            tpfdata.tot_grad_norm += norm.item() / args.p_step
            optimizer.step()
            lr_scheduler.step()
            
            cur_reward = torch.mean(state_dict["final_reward"]).item()
    
            print(f"count: [{count} / {args.max_train_steps // args.p_step}]")
            print("train_reward", cur_reward)
            print("grad norm", tpfdata.tot_grad_norm, "ratio", tpfdata.tot_ratio)
            print("kl", tpfdata.tot_kl, "p_loss", tpfdata.tot_p_loss)
                
            wandb.log({"train_reward": cur_reward,
                       "grad norm": tpfdata.tot_grad_norm,
                       "ratio": tpfdata.tot_ratio,
                       "kl": tpfdata.tot_kl,
                       "p_loss": tpfdata.tot_p_loss,
                       "lr": optimizer.state_dict()['param_groups'][0]['lr']
                       }, step=count)
            
        if cur_reward > best_reward:
            best_reward = cur_reward   
            ckpt_path_best = os.path.join(args.save_ckpt_path, '%d.pt' % (count))   
            torch.save({
                        'config': config,
                        'model': moldiff.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        'iteration': count,
                    }, ckpt_path_best)

        if count == (args.max_train_steps // args.p_step) - 1:  
            ckpt_path_last = os.path.join(args.save_ckpt_path, '%d.pt' % (count))  
            torch.save({
                        'config': config,
                        'model': moldiff.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        'iteration': count,
                    }, ckpt_path_last)
        torch.cuda.empty_cache()
