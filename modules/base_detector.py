import abc
import torch
import torch.nn as nn
from typing import Union

class AbstractDetector(nn.Module, metaclass=abc.ABCMeta):
    """
    All deepfake detectors should subclass this class.
    """
    def __init__(self, load_param: Union[bool, str] = False):
        """
        load_param:  (False | True | Path(str))
            False Do not read; True Read the default path; Path Read the required path
        """
        super().__init__()

    @abc.abstractmethod
    def features(self, data_dict: dict) -> torch.tensor:
        """
        Returns the features from the backbone given the input data.
        """
        pass

    @abc.abstractmethod
    def forward(self, data_dict: dict, inference=False) -> dict:
        """
        Forward pass through the model, returning the prediction dictionary.
        """
        pass

    @abc.abstractmethod
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """
        Classifies the features into classes.
        """
        pass

    @abc.abstractmethod
    def build_backbone(self, config):
        """
        Builds the backbone of the model.
        """
        pass
