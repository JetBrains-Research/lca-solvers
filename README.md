# Training Pipeline

## Description

This repository provides a pipeline for fine-tuning a one-line code completion 
language model to study its ability to adapt to different types of composers in 
different hyperparameter settings. PyTorch was chosen as the basis. At the moment, 
only the model and its tokenizer are used from the Hugging Face ecosystem.

## Get started

### Installation

1. Clone this branch to your machine
2. Configure the virtual environment
3. Install all dependencies (see `requirements.txt`)

### Creating runs

Before creating a run, you may want to select configuration files other than the 
standard ones. All available options are located in the `configs` folder. The 
`runs/.templates` folder also contains possible variants for the `run.py` file.

```bash
python3 -m pipeline <your-run-name>
```

You can get a bit more information by running the following command line and 
reading the <a href="#features">Features</a> section.

```bash
python3 -m pipeline --help
```

### Launching

Before running, you can customize your selected configuration files by editing them in the 
`runs/<your-run-name>/configs` folder. The contents of the `configs` folder in the project root 
**should not** be changed for this purpose.

You can also extend and customize the contents of the `run.py` file to fine-tune an individual run. 
And, if necessary, add your changes globally to the original `pipeline` code.

```bash
source <your-python-environment>
export PYTHONPATH=~/lca-solvers  # or other path to this project that is relevant to you
cd runs/<your-run-name>
python3 run.py
```

## Features

### Structure

```bash
   .
   ├── ...
   ├── configs                        # default configuration files (see pipeline/configs)
   ├── pipeline                       # main classes and functions
   └── runs                           # directory containing individual instances of runs
        ├── .templates                # some basic run.py file options to start with
        └── <your-run-name>           # directory of a run
            ├── checkpoints           # filled in as run.py runs
            │   ├── ...
            │   └── 0009              # iteration number when the checkpoint was made
            │       ├── metrics.json  # is used for checkpoint comparison
            │       ├── model         # the result of HF .save_pretrained() call
            │       └── optim.pt      # optimizer state dict
            ├── configs               # folder with (modified) copies of the main .yaml files
            ├── logs                  # filled in as run.py runs
            │   ├── stderr.json       # stderr organized as .json
            │   ├── stdout.json       # stdout organized as .json
            │   ├── train.csv         # training metrics
            │   └── valid.csv         # validation metrics
            ├── run.py                # executable script for fine-tuning
            └── wandb                 # W&B derivative directory
```

To understand how all the customization is divided into different config files, 
you can look at their code in `pipeline/configs` and examples from `configs`.

### Data splitting

The **train_test_split** function from `pipeline/data/dataset.py` is designed 
to split a dataset into disjoint train and test (validation) sets.

### Data preparation

The data is prepared in two steps.

#### Composers

Composers are designed to merge all snapshot files except the completion file into 
one. To add your own composer, simply inherit from the **ComposerBase** class and implement 
the **compose_context** method.

**GrainedComposer** adds a small level of abstraction to define methods for slicing the 
context into chunks (**chunk_datapoint**) and combining them (**combine_chunks**).

**RankingComposer** uses the **GrainedComposer** setting to reorganize chunks in ascending 
order of a given **ranking_function** before combining them. **PathDistanceComposer** 
illustrates this use case.

The related subpackage is `pipeline/data/composers`.

#### Preprocessors

Preprocessors merge the context and completion parts together for further tokenization. 
In addition to the usual **input_ids** and **target_ids** fields, the output also contains 
**loss_mask** to filter out unnecessary model outputs that should not be trained on, 
and **category_ids** to group tokens into categories from the [LCA paper](https://arxiv.org/abs/2406.11612):

- commited,
- common,
- infile,
- inproject,
- non-informative,
- random.

This can be useful for further calculating metrics in each category separately.

**LMPreprocessor** is an example of how the described functionality can be implemented. **loss_mask** 
is filled with ones with respect to the entire length of the context, starting from its end. 
Since HF tokenizers do not have the ability to stop after reaching the maximum number of tokens,
most of the context is truncated in advance. This reduces the runtime of the tokenizer. **context_tokens** 
is used to indicate the minimum number of tokens taken from the context, it can be specified
either in absolute values (**int**) or in relative values (**float**).

The related subpackage is `pipeline/data/preprocessing`.

### Model selection

If some model initialization parameters are not specified in the model configuration file, 
the pipeline will automatically select the best possible option for your hardware.

The related subpackage is `pipeline/model`.

### Trainers

The **FullFineTuningTrainer** is implemented, which supports checkpointing, logging and 
gradient accumulation.

**AdamW** was chosen as the optimizer. The scheduler is implemented by the explicit function 
**get_lr_from_cosine_scheduler_with_linear_warmup** in `pipeline/trainers/utils/schedulers.py`.

**FusedSampler** from `pipeline/trainers/utils/fused_sampler.py` allows the training process to be 
resumed deterministically from its termination point without having to iterate through passed data points.

Currently, only the single GPU training option is supported.

The related subpackage is `pipeline/trainers`.

### Checkpointing

In addition to the model itself, the state of the optimizer and metrics are also saved. 
This allows you to resume training from the middle if necessary.

The related module is `pipeline/outputs/checkpointing.py`.

### Logging

**DummyLogger** is needed as a placeholder in case of debugging.

**LocalLogger** creates the logging structure described in the <a href="#structure">Structure</a> 
section above. The **train_csv** and **valid_csv** files must be different. **stdout_file** and **stderr_file** 
can be the same.

**WandbLogger** wraps **LocalLogger** to additionally send logs to [W&B](https://jetbrains.wandb.io/machine-learning-methods-in-software-engineering/LCA%20Context%20Composers/).

The related subpackage is `pipeline/outputs/loggers`.

## License

MIT
