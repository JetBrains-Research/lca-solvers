from abc import ABC, abstractmethod

import torch


class TrainerBase(ABC):  # TODO: share functionality
    pass
    # def __init__(self) -> None:
    #     # self.skip_vali
    #     self.train_dl = None
    #     self.valid_dl = None
    #
    # @abstractmethod
    # @torch.inference_mode
    # def validate(self) -> ...:  # TODO
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def train(self) -> ...:  # TODO
    #     raise NotImplementedError
