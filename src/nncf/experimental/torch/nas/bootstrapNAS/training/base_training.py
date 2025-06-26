# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC
from abc import abstractmethod

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor
from nncf.torch.compression_method_api import PTCompressionAlgorithmController


class BNASTrainingAlgorithm(ABC):
    """
    Base training algorithm for supernet-based NAS methods.
    """

    @abstractmethod
    def step(self) -> None:
        """
        Should be called at the beginning of each training step for activation some Subnet(s).
        """

    @abstractmethod
    def set_stage(self, stage_desc: StageDescriptor) -> None:
        """
        Set a new training stage with parameters from a given stage descriptor

        :param stage_desc: describes parameters of the training stage that should be enabled
        """

    @abstractmethod
    def prepare_for_validation(self) -> None:
        """
        Performs some action on active subnet or supernet before validation. For instance, it can be the batchnorm
        adaptation to achieve the best accuracy on validation.
        """

    @abstractmethod
    def get_total_num_epochs(self) -> int:
        """
        Returns total number of epochs required for the supernet training.

        :return: number of epochs
        """


class BNASTrainingController(PTCompressionAlgorithmController, BNASTrainingAlgorithm, ABC):
    """
    A base class for BootstrapNAS training controllers that provides capabilities for supernet training.
    """

    @property
    @abstractmethod
    def multi_elasticity_handler(self) -> MultiElasticityHandler:
        """
        Gets access to multi elasticity handler to perform some actions with supernet or subnets.

        :return: multi elasticity handler
        """

    @property
    @abstractmethod
    def elasticity_controller(self) -> ElasticityController:
        """
        Gets access to elasticity controller. Usually it's needed for saving its state for further resuming in the
        search part.

        :return: elasticity controller
        """
