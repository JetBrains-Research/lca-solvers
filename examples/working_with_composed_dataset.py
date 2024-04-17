from data_classes.datapoint_composed import DatapointComposed
from data_loading.composer_train_dataset_loading import DataLoaderTrainComposed


def compose_input_sequence(dp: DatapointComposed, max_len: int, context_ratio: float = 0.75) -> str:
    if not -1e8 < context_ratio < 1. + 1e8:
        raise ValueError('context_ratio must be between 0 and 1')
    context = dp.context
    completion = dp.completion

    length_context = int(max_len * context_ratio) + 1
    length_completion = int(max_len * (1 - context_ratio)) + 1

    compl_trim_idx = completion.find('\n', length_completion)
    context_trim_idx = context.rfind('\n', 0, -length_context)

    if compl_trim_idx > 0:
        completion_trimmed = completion[:compl_trim_idx]
    else:
        completion_trimmed = completion

    if context_trim_idx > 0:
        context_trimmed = context[1+context_trim_idx:]
    else:
        context_trimmed = context

    return context_trimmed + '\n---------CONTEXT SEPARATOR---------\n' + completion_trimmed


def main():
    loader = DataLoaderTrainComposed(
        hf_path='JetBrains-Research/context-py-train',
        hf_config=None,
        cache_dir='/mnt/data2/shared-data/lca/hf_cache/',
    )
    for idx, dp_out in enumerate(loader.get_datapoint_iterator()):
        dp = DatapointComposed.from_hf_datapoint(dp_out.dp)
        input_seq = compose_input_sequence(dp, max_len=500, context_ratio=0.75)
        print(input_seq)
        if idx > 2:
            break


if __name__ == '__main__':
    main()
