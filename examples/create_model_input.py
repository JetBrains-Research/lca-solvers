from context_composers.context_composer_example import ContextComposerExample
from data_classes.datapoint_py import DatapointPy
from data_processing.dataset_loading import DataLoader


def main() -> None:
    loader = DataLoader()
    composer = ContextComposerExample()
    for idx, hf_dp in enumerate(loader.get_dp_iterator(1)):
        dp = DatapointPy.from_hf_datapoint(hf_dp)
        print(composer.compose_context(dp))
        print('\n', '=' * 50, '\n')
        if idx > 1:
            break

if __name__ == '__main__':
    main()