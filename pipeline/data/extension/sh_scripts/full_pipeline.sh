lca_dir='/mnt/data/shared-data/lca/plcc_data_dir/'
language='kotlin'
workers=16

cd ..

echo "Extracting Completion Files from Repositories"
python -m repo_processing.scripts.collect_completion_files \
  --lca-dir "$lca_dir" \
  --language "$language" \
  -e '.kt' \
  -w $workers

echo "Extracting Repository Snapshots"
python -m repo_processing.scripts.collect_repo_snapshots_first_run \
  --lca-dir "$lca_dir" \
  --language "$language" \
  -e '.kt' \
  -w $workers

echo "Adding Extra Metainfo"
python -m repo_processing.scripts.add_snapshot_hash_to_commit_metadata \
  --lca-dir "$lca_dir" \
  --language "$language"

echo "Line Classification"
python -m line_classification.scripts.classify_lines \
  --lca-dir "$lca_dir" \
  --language "$language" \
  --strategy "dummy"

echo "Completion Files to Parquet"
python -m hf_processing.scripts.completion_files_to_parquet \
  --lca-dir "$lca_dir" \
  --language "$language" \
  --max-lines 2000

#python -m hf_processing.scripts.repo_snapshots_to_parquet \
#  --lca-dir "$lca_dir" \
#  --language "$language"
