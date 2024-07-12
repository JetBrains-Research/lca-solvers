from torch.utils.data import Dataset, DataLoader
from huggingface_hub import notebook_login
import torch
import pickle
import sys
sys.path.append('/home/blatova/lca-solvers/')

from datasets import load_dataset


notebook_login()

class HFDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        return self.hf_dataset[idx]

dataset = load_dataset("JetBrains-Research/context-py-train", "path_distance_relevant", cache_dir='/mnt/data2/shared-data/lca/hf_cache2/')

def get_new_val_dataset(num_points = 128):
    curr_num_points = 0
    validation_repos = set()
    train_repos = list(set(dataset['train']['repo']))
    cnt = Counter(dataset['train']['repo'])
    val_indices_list = []
    while curr_num_points < num_points:  
        curr_repo = random.choice(train_repos)
        num_points_to_add = min(5, num_points-curr_num_points, cnt[curr_repo])
        indices = [i for i, value in enumerate(dataset['train']['repo']) if value == curr_repo]
        val_indices_list.extend(random.sample(indices, num_points_to_add))
        validation_repos.add(curr_repo)
        train_repos.remove(curr_repo)
        curr_num_points = curr_num_points + num_points_to_add
    val_dataset = dataset['train'].select(val_indices_list)
    return val_dataset, validation_repos, val_indices_list

def get_standard_val_dataset(val_indices_list):
    return dataset['train'].select(val_indices_list)

def write_val_dataset(std_val_dataset_version, val_dataset, validation_repos, val_indices_list):
    torch.save(val_dataset, f'std_val_dataset/std_val_dataset_{std_val_dataset_version}.pt')    
    with open(f"std_val_dataset/std_val_dataset_{std_val_dataset_version}_repos.pkl", "wb") as repos: 
        pickle.dump(validation_repos, repos)
    with open(f"std_val_dataset/std_val_dataset_{std_val_dataset_version}_indices.pkl", "wb") as indices: 
        pickle.dump(val_indices_list, indices)
    val_dataset.push_to_hub(f"val_dataset_std_{std_val_dataset_version}")
    

def read_val_dataset(std_val_dataset_version):
    with open(f"std_val_dataset/std_val_dataset_{std_val_dataset_version}_repos.pkl", "rb") as repos: 
        validation_repos = pickle.load(repos)
    with open(f"std_val_dataset/std_val_dataset_{std_val_dataset_version}_indices.pkl", "rb") as indices: 
        val_indices_list = pickle.load(indices)
    val_dataset = get_standard_val_dataset(val_indices_list)
    return val_dataset, validation_repos, val_indices_list

def is_validation(entry, validation_repos):
    return entry['repo'] in validation_repos

def is_train(entry, validation_repos):
    return entry['repo'] not in validation_repos

def get_new_ema(curr_loss, prev_ema):
    return (curr_loss - prev_ema)*EMA_ALPHA + prev_ema

def get_val_dataset(use_standard_dataset, save_standard_dataset, std_val_dataset_version):    
    if (not use_standard_dataset):
        val_dataset, validation_repos, val_indices_list = get_new_val_dataset() 
        if save_standard_dataset:
            write_val_dataset(std_val_dataset_version, val_dataset, validation_repos, val_indices_list)
        return val_dataset, validation_repos, val_indices_list
    else:
        val_dataset, validation_repos, val_indices_list = read_val_dataset(std_val_dataset_version) 
        return val_dataset, validation_repos, val_indices_list
              
        

def get_train_and_val_loader(use_standard_dataset, save_standard_dataset, std_val_dataset_version):
    val_dataset, validation_repos, val_indices_list = get_val_dataset(use_standard_dataset, save_standard_dataset, std_val_dataset_version)
    train_dataset = dataset['train'].filter(lambda x: is_train(x, validation_repos))
    torch_train_dataset = HFDataset(train_dataset)
    train_loader = DataLoader(torch_train_dataset, batch_size=1, shuffle=True)
    
    torch_val_dataset = HFDataset(val_dataset)
    val_loader = DataLoader(torch_val_dataset, batch_size=1, shuffle=True)
    return train_loader, val_loader
    






