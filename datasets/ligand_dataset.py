import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
from utils.data import PDBProtein, parse_sdf_file
from datasets.pl_data import ProteinLigandData, torchify_dict
from utils.evaluation.scoring_func import get_chem
from rdkit import Chem

class LigandDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='multiprop'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path), 'name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, ligand_fn in enumerate(tqdm(index)):
                try:
                    data_prefix = self.raw_path
                    print(os.path.join(data_prefix, ligand_fn))
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    mol = Chem.MolFromMolFile(os.path.join(data_prefix, ligand_fn))
                    chem_results = get_chem(mol)
                    data.properties = [chem_results['qed'], chem_results['sa'], chem_results['logp'], 
                                        chem_results['tpsa'], chem_results['hba'], chem_results['lipinski'],
                                        chem_results['hbd'], chem_results['fsp3'], chem_results['rotb']]
                    print(data.properties)
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['ligand_filename'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)
        
    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))

        data = ProteinLigandData(**data)
        data.id = idx

        return data
        

if __name__ == '__main__':

    path = '/DATA2/east/molecule3D/molecule3D/'
    dataset = LigandDataset(path)
    print(len(dataset), dataset[0])