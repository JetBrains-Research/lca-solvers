import abc

import torch


class TrainerBase(abc.ABC):  # TODO: share functionality
    pass
    # def __init__(self) -> None:
    #     # self.skip_vali
    #     self.train_dl = None
    #     self.valid_dl = None
    #
    # @abc.abstractmethod
    # @torch.inference_mode
    # def validate(self) -> ...:  # TODO
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def train(self) -> ...:  # TODO
    #     raise NotImplementedError
