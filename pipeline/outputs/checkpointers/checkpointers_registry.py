from pipeline.outputs.checkpointers.checkpointer import CheckpointManager
from pipeline.outputs.checkpointers.top_k_checkpointer import TopKCheckpointManager

CHECKPOINTERS_REGISTRY = {
    'checkpointer': CheckpointManager,
    'top_k_checkpointer': TopKCheckpointManager,
}

