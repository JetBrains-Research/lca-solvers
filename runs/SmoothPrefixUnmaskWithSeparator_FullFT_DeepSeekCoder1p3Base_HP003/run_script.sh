/home/sapronov/.virtualenvs/lca-solvers/bin/python3 \
/home/sapronov/lca-solvers/pipeline/__main__.py \
run_name=SmoothPrefixUnmaskWithSeparator_FullFT_DeepSeekCoder1p3Base_HP003 \
adapter=smooth_prefix_unmask_adapter/medium_decay \
composer=chained_composer/python_files \
preprocessor=completion_loss_preprocessor/full_completion_loss_with_sep_16k \
trainer=full_finetuning_trainer/past_weights