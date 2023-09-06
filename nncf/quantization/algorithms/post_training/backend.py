# Copyright (c) 2023 Intel Corporation
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
from pathlib import Path
from typing import Any, Dict, List, Tuple

from nncf import Dataset
from nncf.quantization.algorithms.post_training.algorithm import TModel


class PostTrainingBackend(ABC):
    @abstractmethod
    def make_tasks(
        self, model: TModel, calibration_dataset: Dataset, subset_size: int
    ) -> List[Tuple[TModel, Dataset, Dict[str, Any]]]:
        """
        Returns model subgraphs and the datasets for calibrations for particular model.

        :param model: Model from which the model subgraphs are obtained.
        :param calibration_dataset: Calibration dataset for original model.
        :param subset_size: Number of samples used to get new dataset.
        :return: All quantization tasks from particular model.
        """

    @abstractmethod
    def is_single_model(self, model: TModel) -> bool:
        """
        Chechks whether a model has inner subgraphs to quantize.

        :param model: Model to check.
        :return: True if the model has no inner subgraphs, otherwise - False.
        """

    @abstractmethod
    def dump_model(self, model: TModel, dir: Path, backend_params: Dict[str, Any]) -> None:
        """
        Save a model to a directory. Backend params are used to determine the model name to dump.

        :param model: Model to dump.
        :param dir: Directory path.
        :param backend_params: Backend specific parameters.
        """

    @abstractmethod
    def set_subgraph(self, subgraph_model: TModel, backend_params: Dict[str, Any]) -> None:
        """
        Set subgraph model to an original model.

        :param subgraph_model: Model to set.
        :param backend_params: Backend specific parameters.
        """
