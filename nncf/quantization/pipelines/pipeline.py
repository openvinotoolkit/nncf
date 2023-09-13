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
from typing import TypeVar

from nncf.data.dataset import Dataset

TModel = TypeVar("TModel")


class Pipeline(ABC):
    """
    A base class for creating pipelines that apply algorithms to a model.

    This abstract class serves as an interface for creating custom model
    processing pipelines that encapsulate a series of algorithms to be
    applied to a model using a provided dataset.
    """

    @abstractmethod
    def run(self, model: TModel, dataset: Dataset) -> TModel:
        """
        Abstract method that defines the sequence of algorithms to be
        applied to the provided model using the provided dataset.

        :param model: A model to which pipeline will be applied.
        :param dataset: A dataset that holds the data items for algorithms.
        """
