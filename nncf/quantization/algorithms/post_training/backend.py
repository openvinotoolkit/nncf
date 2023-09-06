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
from typing import Any, Dict, Iterable, List, Tuple

from nncf import Dataset
from nncf.data.dataset import DataItem
from nncf.quantization.algorithms.post_training.algorithm import TModel


class PostTrainingBackend(ABC):
    @staticmethod
    @abstractmethod
    def collect_dataitems_for_children_models(
        model: TModel, calibration_dataset: Dataset, subset_size: int, model_cnt: int
    ) -> Iterable[DataItem]:
        """
        Returns dataitems for children models of the main model.

        :param model: Model to infer to collect dataitems.
        :param calibration_dataset: Dataset is used to collect new dataitems.
        :param subset_size: Size of dataitems to collect
        :param model_cnt: Global model number.
        """

    @staticmethod
    @abstractmethod
    def make_dataset_for_child_models(dataitems: Iterable[DataItem], backend_params: Dict[str, Any]) -> Dataset:
        """
        Return dataset for child models.

        :param dataitems: Data items to collect into dataset.
        :param backend_params: Backend-specific parameters.
        """

    @staticmethod
    @abstractmethod
    def is_single_model(model: TModel) -> bool:
        """
        Chechks whether a model has inner subgraphs to quantize.

        :param model: Model to check.
        :return: True if the model has no inner subgraphs, otherwise - False.
        """

    @staticmethod
    @abstractmethod
    def get_child_models(model: TModel) -> List[Tuple[TModel, Dict[str, Any]]]:
        """
        Returns all child models of passed model.

        :param model: Model to seek for child models.
        :return: Models with backend specific parameters.
        """

    @staticmethod
    @abstractmethod
    def add_additional_outputs(model: TModel) -> TModel:
        """
        Returns the model with additional outputs to collect statistics for child models.

        :param model: Model to update.
        :return: Updated model with extra outputs.
        """

    @staticmethod
    @abstractmethod
    def dump_model(model: TModel, dir: Path, backend_params: Dict[str, Any]) -> None:
        """
        Save a model to a directory. Backend params are used to determine the model name to dump.

        :param model: Model to dump.
        :param dir: Directory path.
        :param backend_params: Backend specific parameters.
        """

    @staticmethod
    @abstractmethod
    def set_child_model(child_model: TModel, backend_params: Dict[str, Any]) -> None:
        """
        Set subgraph model to an original model.

        :param subgraph_model: Model to set.
        :param backend_params: Backend specific parameters.
        """
