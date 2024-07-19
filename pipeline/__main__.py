from pipeline.environment.run_directory import *

import argparse
import os
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Creates new run directory from predefined templates.',
    )
    copy_triplets = (
        [SRC_RUN_PY, RUNS_SRC_DIR, RUN_PY],
        [SRC_CHECKPOINTING_YAML, CHECKPOINTING_SRC_DIR, CHECKPOINTING_YAML],
        [SRC_COMPOSER_YAML, COMPOSERS_SRC_DIR, COMPOSER_YAML],
        [SRC_DATASET_YAML, DATASETS_SRC_DIR, DATASET_YAML],
        [SRC_LOGGER_YAML, LOGGERS_SRC_DIR, LOGGER_YAML],
        [SRC_MODEL_YAML, MODELS_SRC_DIR, MODEL_YAML],
        [SRC_PREPROCESSOR_YAML, PREPROCESSORS_SRC_DIR, PREPROCESSOR_YAML],
        [SRC_SPLIT_YAML, SPLITS_SRC_DIR, SPLIT_YAML],
        [SRC_TRAINER_YAML, TRAINERS_SRC_DIR, TRAINER_YAML],
    )

    parser.add_argument(
        'name',
        type=str,
        help='Unique name of a run. Hereinafter referred to as <name>',
    )
    for src_file, src_dir, dest_file in copy_triplets:
        parser.add_argument(
            f'--src-{os.path.basename(dest_file).replace(".", "-")}',
            type=str,
            default=src_file,
            choices=os.listdir(src_dir),
            help=f'Initial content of a {os.path.join(RUNS_DIR, "<name>", dest_file)}. You can chose from predefined '
                 f'templates in the {src_dir} directory',
        )

    args = parser.parse_args()
    run_dir = os.path.join(RUNS_DIR, args.name)

    if os.path.exists(run_dir):
        raise ValueError(f'Directory {run_dir} already exists.')

    os.mkdir(run_dir)

    for dest_dir in (CHECKPOINTS_DIR, LOGS_DIR):
        dest_dir = os.path.join(run_dir, dest_dir)
        os.mkdir(dest_dir)
        os.mknod(os.path.join(dest_dir, '.gitkeep'))

    os.mkdir(os.path.join(run_dir, CONFIGS_DIR))
    for src_file, (_, src_dir, dest_file) in zip(list(vars(args).values())[1:], copy_triplets):
        src = os.path.join(src_dir, src_file)
        dest = os.path.join(run_dir, dest_file)
        shutil.copyfile(src, dest)


if __name__ == '__main__':
    main()
