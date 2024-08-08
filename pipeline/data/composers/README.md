## Structure

1. `file_filtering`
2. `file_preprocessing`
3. `file_chunking`
4. `chunk_ranking`
5. `chunk_sorting`
6. `chunk_harvesting`
7. `context_postprocessing`

See `extra/attachments/chained_composer.png` for block dependencies info.

## Examples

First of all, set the environment: 

```bash
source <your-python-environment>
export PYTHONPATH=~/lca-solvers  # or other path to this project that is relevant to you
```

Default run:

```bash
python3 -m pipeline run_name=YourAwesomeRunName
```

In case of debugging:

```bash
python3 -m pipeline run_name=YourAwesomeRunName logger=local/local preprocessor=completion_loss_preprocessor/debugging trainer=full_finetuning_trainer/debugging
```

New composer:

```bash
python3 -m pipeline run_name=YourAwesomeRunName composer=chained_composer/your_new_composer_config
```
