from data_processing.context_composers.context_composer_example import ContextComposerExample
from data_processing.data_classes.datapoint_py import DatapointPy
from data_processing.data_filters.repo_snapshot_filter_stack import SnapshotFilterStack
from data_processing.data_loading.raw_train_dataset_loading import DataLoaderTrainRaw


def main() -> None:
    loader = DataLoaderTrainRaw()
    composer = ContextComposerExample()
    filt = SnapshotFilterStack()
    for idx, hf_dp in enumerate(loader.get_dp_iterator(1)):
        dp = DatapointPy.from_hf_datapoint(hf_dp, filtering=filt)
        print(composer.compose_context(dp))
        print('\n', '=' * 50, '\n')
        if idx > 2:
            break


if __name__ == '__main__':
    main()
