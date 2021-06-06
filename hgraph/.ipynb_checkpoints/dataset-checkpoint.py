import torch
from torch.utils.data import Dataset
from rdkit import Chem
import os, random, gc
import pickle

from hgraph.chemutils import get_leaves
from hgraph.mol_graph import MolGraph
import cv2

class MoleculeDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size):
        safe_data = []
        for mol_s in data:
            hmol = MolGraph(mol_s)
            ok = True
            for node,attr in hmol.mol_tree.nodes(data=True):
                smiles = attr['smiles']
                ok &= attr['label'] in vocab.vmap
                for i,s in attr['inter_label']:
                    ok &= (smiles, s) in vocab.vmap
            if ok: 
                safe_data.append(mol_s)

        print(f'After pruning {len(data)} -> {len(safe_data)}') 
        self.batches = [safe_data[i : i + batch_size] for i in range(0, len(safe_data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)


class MolEnumRootDataset(Dataset):

    def __init__(self, data, vocab, avocab):
        self.batches = data
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.batches[idx])
        leaves = get_leaves(mol)
        smiles_list = set( [Chem.MolToSmiles(mol, rootedAtAtom=i, isomericSmiles=False) for i in leaves] )
        smiles_list = sorted(list(smiles_list)) #To ensure reproducibility

        safe_list = []
        for s in smiles_list:
            hmol = MolGraph(s)
            ok = True
            for node,attr in hmol.mol_tree.nodes(data=True):
                if attr['label'] not in self.vocab.vmap:
                    ok = False
            if ok: safe_list.append(s)
        
        if len(safe_list) > 0:
            return MolGraph.tensorize(safe_list, self.vocab, self.avocab)
        else:
            return None


class MolPairDataset(Dataset):

    def __init__(self, data, vocab, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y = zip(*self.batches[idx])
        x = MolGraph.tensorize(x, self.vocab, self.avocab)[:-1] #no need of order for x
        y = MolGraph.tensorize(y, self.vocab, self.avocab)
        return x + y

class DataFolder(object):

    def __init__(self, data_folder, batch_size, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data_files) * 1000

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            if self.shuffle: random.shuffle(batches) #shuffle data before batch
            for batch in batches:
                yield batch

            del batches
            gc.collect()

from sklearn.utils import shuffle as skutils 
class DataFolder_BMS(object):

    def __init__(self, data_folder, batch_size, transform, path_img, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.path_img = path_img

    def __len__(self):
        return len(self.data_files) * 50

    def __iter__(self):
        for fn_t in self.data_files:
            fn = os.path.join(self.data_folder, fn_t)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)
                
            fn_ids = os.path.join('/home/quang/working/Theory_of_ML/hgraph2graph/train_processed_bms/ids', fn_t)
            with open(fn_ids, 'rb') as f:
                batches_ids = pickle.load(f)

            if self.shuffle: 
#                 random.shuffle(batches) #shuffle data before batch
                batches, batches_ids = skutils(batches, batches_ids)
            
            for batch, batch_ids in zip(batches, batches_ids):
                images_list = []
                for id_temp in batch_ids:
                    image = cv2.imread(os.path.join(self.path_img, id_temp + '.png'))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    transformed = self.transform(image=image)
                    image = transformed['image']
                    images_list.append(image)
                batch_image = torch.stack(images_list)
                yield batch, batch_image

            del batches, batch_image, batches_ids
            gc.collect()
            
            
class DataFolder_ver2_BMS(object):

    def __init__(self, data_folder, ids_path, batch_size, transform, path_img, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.path_img = path_img
        self.ids_path = ids_path

    def __len__(self):
        return len(self.data_files) * 50

    def __iter__(self):
        for fn_t in self.data_files:
            fn = os.path.join(self.data_folder, fn_t)
            with open(fn, 'rb') as f:
                batches = pickle.load(f)
                
            fn_ids = os.path.join(self.ids_path, fn_t)
            with open(fn_ids, 'rb') as f:
                batches_ids, batches_labels = pickle.load(f)

            if self.shuffle: 
#                 random.shuffle(batches) #shuffle data before batch
                batches, batches_ids, batches_labels = skutils(batches, batches_ids, batches_labels)
            
            for batch, batch_ids, batches_label in zip(batches, batches_ids, batches_labels):
                images_list = []
                for id_temp in batch_ids:
                    image = cv2.imread(os.path.join(self.path_img, id_temp + '.png'))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    transformed = self.transform(image=image)
                    image = transformed['image']
                    images_list.append(image)
                batch_image = torch.stack(images_list)
                yield batch, batch_image, batches_label

            del batches, batch_image, batches_ids, batches_labels
            gc.collect()


