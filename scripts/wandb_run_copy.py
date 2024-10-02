"""
Related issues:
- https://github.com/wandb/wandb/issues/4790
- https://github.com/wandb/wandb/issues/5726
"""

import pandas as pd
import wandb

wandb.login()
api = wandb.Api()

src_entity = 'machine-learning-methods-in-software-engineering'
src_project = 'LCA Context Composers'

dest_entity = 'machine-learning-methods-in-software-engineering'
dest_project = 'LCA Turrets'

names = {  # scr -> dest
    'FileLevel_FullFT_DeepSeekCoder1p3Base_HP002': 'file_level_baseline',
    'PythonFiles_FullFT_DeepSeekCoder1p3Base_HP002': 'python_files_composer_baseline',
}

runs = api.runs(f'{src_entity}/{src_project}')

for run in runs:
    if run.name in names:
        history = run.history(samples=run.lastHistoryStep + 1)
        files = run.files()

        new_run = wandb.init(
            project=dest_project,
            entity=dest_entity,
            config=run.config,
            name=names[run.name],
            resume='allow',
        )

        for index, row in history.iterrows():
            assert index == row['_step']

            log = {
                k: v if v != 'NaN' else float('nan')
                for k, v in row.to_dict().items()
                if v is not None and not pd.isna(v)
            }

            new_run.log(log, step=index)

        for file in files:
            file.download(replace=True)
            new_run.save(file.name, policy="now")

        new_run.finish()
