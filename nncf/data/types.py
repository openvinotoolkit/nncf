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
from typing import TypeVar

# Contains type aliases that are used inside the `nncf.data`.

# This is a type alias for the element of the data source.
# For example, the `DataItem` object can be the tensor-like object,
# dict of tensor-like objects and etc. Usually it contains both samples for
# inference and targets.
DataItem = TypeVar('DataItem')


# This is a type alias for the user's data source. We assume it is
# an [iterable](https://docs.python.org/3/glossary.html#term-iterable) python object.
# It means that when the `DataSource` object is passed as an argument to the built-it
# function `iter()`, it returns an iterator for the `DataSource` object.
DataSource = Iterable[DataItem]


# This is a type alias for the model's input. It means that we can run an inference of the model
# for the `ModelInput` object. Usually `ModelInput` is a part of the `DataItem` object.
ModelInput = TypeVar('ModelInput')
