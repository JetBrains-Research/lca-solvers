/home/sapronov/.virtualenvs/lca-solvers/bin/python3 \
/home/sapronov/lca-solvers/pipeline/__main__.py \
run_name=SmoothPrefixUnmaskWithSeparator_FullFT_DeepSeekCoder1p3Base_HP002 \
adapter=smooth_prefix_unmask_adapter/high_decay \
composer=chained_composer/python_files \
preprocessor=completion_loss_preprocessor/full_completion_loss_with_sep_16k \
trainer=universal_trainer/past_weights