# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from torch.utils.data import DataLoader

from nncf.common.initialization.dataloader import NNCFDataLoader


class PTInitializingDataLoader(NNCFDataLoader):
    """
    This class wraps the torch.utils.data.DataLoader class,
    and defines methods to parse the general data loader output to
    separate the input to the compressed model and the ground truth target
    for the neural network. This is required for proper initialization of
    certain compression algorithms.
    """

    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader

    @property
    def batch_size(self):
        return self._data_loader.batch_size

    def __iter__(self):
        return iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader)

    def get_inputs(self, dataloader_output: Any) -> tuple[tuple, dict]:
        """Returns (args, kwargs) for the current model call to be made during the initialization process"""
        raise NotImplementedError

    def get_target(self, dataloader_output: Any) -> Any:
        """
        Parses the generic data loader output and returns a structure to be used as
        ground truth in the loss criterion.

        :param dataloader_output - the (args, kwargs) tuple returned by the __next__ method.
        """
        raise NotImplementedError
