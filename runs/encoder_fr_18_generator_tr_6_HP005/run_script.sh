/home/sapronov/.virtualenvs/lca-solvers/bin/python3 \
/home/sapronov/lca-solvers/pipeline/__main__.py \
run_name=encoder_fr_18_generator_tr_6_HP005 \
adapter=split_adapter/generator_6 \
composer=split_composer/python_files_20k_32 \
logger=wandb_logger/wandb_turrets \
preprocessor=split_completion_loss_preprocessor/full_completion_loss_20k_40k \
+additional_composer=split_composer/python_files_20k_32 \
+additional_preprocessor=split_completion_loss_preprocessor/full_completion_loss_8k_16k