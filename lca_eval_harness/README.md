# LCA Evaluation

This module aims to provide a tool for evaluation of completion tasks.

## Structure

 * [data_classes](./data_classes) – contains datapoint data classes with `split_completion_file`.
functionality to
 * [dataset_loaders](./dataset_loaders) – wrappers for data loading with a dataset accessible in `data` property.
 * [model_inference](./model_inference) – classes for sequence generation with `generate` method.
 * [prompters](./prompters) – different strategies to utilize precomposed context can be implemented here.
 * [tasks](./tasks) – `TaskBase` class has method `run` that do the following:
   * load dataset with [dataset_loaders](./dataset_loaders)
   * use [prompters](./prompters) to create input sequence
   * use generation engine from [model_inference](./model_inference) to generate output sequence
 