from pipeline.outputs.checkpointers.data_structures import Checkpoint
from pipeline.outputs.checkpointers.checkpointer import CheckpointManager

import os
import shutil


class TopKCheckpointManager(CheckpointManager):
    def __init__(self, max_checkpoints_num: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_checkpoints_num = max_checkpoints_num

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        super().save_checkpoint(checkpoint)

        checkpoints = next(os.walk(self.directory))[1]
        checkpoints = sorted(checkpoints, key=self.get_checkpoint_score)

        while len(checkpoints) > self.max_checkpoints_num:
            checkpoint2del = checkpoints.pop()
            checkpoint2del = os.path.join(self.directory, checkpoint2del)
            shutil.rmtree(checkpoint2del)
