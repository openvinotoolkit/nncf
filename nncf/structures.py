"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class NNCFExtraConfigStruct:
    @classmethod
    def get_id(cls) -> str:
        raise NotImplementedError


class QuantizationPrecisionInitArgs(NNCFExtraConfigStruct):
    """
    Stores arguments for initialization of quantization's bitwidth.
    Initialization is based on calculating a measure reflecting layers' sensitivity to perturbations. The measure is
    calculated by estimation of average trace of Hessian for modules using the Hutchinson algorithm.
    :param criterion: loss function, instance of descendant of `torch.nn.modules.loss._Loss`,
    :param data_loader: 'data_loader' - provides an iterable over the given dataset, instance of descendant
                of 'torch.utils.data.DataLoader' class. Must return both inputs and targets to calculate loss
                and gradients.
    :param device: Device to perform initialization at. Either 'cpu' or 'cuda' (default).
    """

    def __init__(self, criterion: _Loss, data_loader: DataLoader, device: str = 'cuda'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device

    @classmethod
    def get_id(cls) -> str:
        return "quantization_precision_init_args"


class QuantizationRangeInitArgs(NNCFExtraConfigStruct):
    """
    Stores arguments for initialization of quantization's ranges.
    Initialization is done by collecting per-layer activation statistics on training dataset in order to choose proper
    output range for quantization.
    :param data_loader: 'data_loader' - provides an iterable over the given dataset, instance of descendant
                of 'torch.utils.data.DataLoader' class. Must return both inputs and targets to calculate loss
                and gradients.
    :param device: Device to perform initialization at. Either 'cpu' or 'cuda' (default).
    """

    def __init__(self, data_loader: DataLoader, device: str = 'cuda'):
        self.data_loader = data_loader
        self.device = device

    @classmethod
    def get_id(cls) -> str:
        return "quantization_range_init_args"


class BNAdaptationInitArgs(NNCFExtraConfigStruct):
    """
    Stores arguments for BatchNorm statistics adaptation procedure.
    Adaptation is done by inferring a number of data batches on a compressed model
    while the BN layers are updating the rolling_mean and rolling_variance stats.
    :param data_loader: 'data_loader' - provides an iterable over the given dataset, instance of descendant
                of 'torch.utils.data.DataLoader' class. Must return both inputs and targets to calculate loss
                and gradients.
    :param device: Device to perform initialization at. Either 'cpu' or 'cuda' (default).
    """

    def __init__(self, data_loader: DataLoader, device: str = 'cuda'):
        self.data_loader = data_loader
        self.device = device

    @classmethod
    def get_id(cls) -> str:
        return "bn_adaptation_init_args"
