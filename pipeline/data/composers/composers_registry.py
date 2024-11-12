from pipeline.data.composers.chained_composer import ChainedComposer
from pipeline.data.composers.split_composer import SplitComposer

COMPOSERS_REGISTRY = {
    'chained_composer': ChainedComposer,
    'split_composer': SplitComposer,
}
