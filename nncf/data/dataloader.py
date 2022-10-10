"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Iterable
from typing import Iterator
from typing import Callable
from abc import ABC
from abc import abstractmethod

from nncf.data.types import DataItem
from nncf.data.types import ModelInput
from nncf.data.types import DataSource


class DataLoader(ABC):
    """
    Describes the interface of the data source that is used by
    compression algorithms.

    The `DataLoader` object contains the dataset and information
    about how to transform the data item returned per iteration to
    the model's expected input.
    """

    @property
    @abstractmethod
    def transform_fn(self) -> Callable[[DataItem], ModelInput]:
        """
        Returns the transformation method that takes a data item
        returned per iteration through the `DataLoader` object and
        transforms it to the model's expected input that can be used
        for the model inference.

        :return: The method that takes data item and transforms it into
            the model's expected input that can be used for the model inference.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[DataItem]:
        """
        Creates an iterator for the data items of this data loader.

        :return: An iterator for the data items of this data loader.
        """


class DataLoaderImpl(DataLoader):
    """
    Implementation of the `DataLoader` for the case when
    the custom data source is the [iterable](https://docs.python.org/3/glossary.html#term-iterable)
    python object.
    """

    def __init__(self,
                 data_source: DataSource,
                 transform_fn: Callable[[DataItem], ModelInput]):
        """
        Initializes the data loader.

        :param data_source: The custom data source that is an iterable
            python object.
        :param transform_fn: Transformation method that takes a data item
            returned per iteration through `DataSource` object and transforms
            it to the model's expected input that can be used for the model inference.
        """
        self._data_source = data_source
        self._transform_fn = transform_fn

    @property
    def transform_fn(self) -> Callable[[DataItem], ModelInput]:
        return self._transform_fn

    def __iter__(self) -> Iterable[DataItem]:
        return iter(self._data_source)
