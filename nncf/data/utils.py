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

from typing import List
from typing import Iterator
from typing import Callable

from nncf.data.types import DataSource
from nncf.data.types import DataItem
from nncf.data.types import ModelInput
from nncf.data.dataloader import DataLoader
from nncf.data.dataloader import DataLoaderImpl


def create_dataloader(data_source: DataSource,
                      transform_fn: Callable[[DataItem], ModelInput]) -> DataLoader:
    """
    Wraps the provided custom data source that is an [iterable](https://docs.python.org/3/glossary.html#term-iterable)
    python object into the NNCF `DataLoader` object.

    :param data_source: Custom data source that is an iterable python object.
    :param transform_fn: The method that takes a data item returned per iteration
        through `DataSource` object and transforms it to the model's expected input
        that can be used for the model inference.
    :return: The object that implements the NNCF `DataLoader` interface and
        wraps custom data source.
    """
    # Checks that `data_source` is an iterable python object.
    try:
        iter(data_source)
    except TypeError as exc:
        raise ValueError('The provided `data_source` object is not an iterable object. Please see '
                         'https://docs.python.org/3/glossary.html#term-iterable to learn more '
                         'about iterable object.') from exc

    return DataLoaderImpl(data_source, transform_fn)


def create_subset(data_loader: DataLoader, indices: List[int]) -> DataLoader:
    """
    Create a new instance of `NNCFDataLoader` that contains only
    specified batches.

    :param data_loader: The data loader to select the specified elements.
    :param indices: The zero-based indices of batches that should be
        selected from provided data loader. The indices should be sorted
        in ascending order.
    :return: The new instance of `NNCFDataLoader` that contains only
        specified batches.
    """
    class BatchSelector:
        def __iter__(self) -> Iterator[DataItem]:
            pos = 0  # Position in the `indices` list.
            num_indices = len(indices)

            for idx, batch in enumerate(data_loader):
                if pos == num_indices:
                    # All specified batches were selected.
                    break
                if idx == indices[pos]:
                    pos = pos + 1
                    yield batch

    return DataLoaderImpl(BatchSelector(), data_loader.transform_fn)
