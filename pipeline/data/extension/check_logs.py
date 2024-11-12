from pathlib import Path
import pandas as pd

logs_dir = Path('/Users/Evgeniy.Glukhov/Datasets/lca/kotlin/_logs/collect_completion_files')

for run_dir in sorted(logs_dir.iterdir()):
    if run_dir.is_dir():
        failed = pd.read_csv(run_dir / 'failed_commits.csv')
        success = pd.read_csv(run_dir / 'successful_commits.csv')
        print(run_dir.name, len(failed), len(success))
