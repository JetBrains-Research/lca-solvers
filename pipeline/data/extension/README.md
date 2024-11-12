# Data Processing Pipeline

## Data Loading
After data is collected you can access the data by:
```python
from data_loading.lca_dataset import LCADataset

lca_dir = '/mnt/data/shared-data/lca/plcc_data_dir/'
language = 'python'  # or 'kotlin' at the moment

ds = LCADataset(lca_dir=lca_dir, language=language, repo_snapshots_source='none')
```

`repo_snapshots_source` parameter could be one of the following:
 - `"none"` – always returns an empty list as a repository snapshot, 
 - `"json"` – reads repository snapshots from `json` files, 
 - `"parquet"` – reads repository snapshots from `parquet` files.


## Run the pipeline
#### WARNING: be careful with overwriting the collected data

1. Copy repositories as in [this notebook](copy_repos.ipynb).
2. The main idea of how to collect the data is in [this script](sh_scripts/full_pipeline.sh).

#### further work:
 - [ ] Automate this script with Hydra
 - [ ] Add Code Engine classification
 - [ ] Add `Update Data` step (for the situation when repositories are updated)

## How the data is stored
Refer to the [LCAFilesystem class](lca_filesystem/lca_filesystem.py) to get the main idea behind the data storage.
