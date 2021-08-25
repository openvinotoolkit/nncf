"""
 Copyright (c) 2021 Intel Corporation
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
import numpy as np

from abc import ABC
from abc import abstractmethod
from typing import Callable, List, TypeVar, Union

TensorType = TypeVar('TensorType')


class BaseTensorListComparator(ABC):
    @classmethod
    @abstractmethod
    def _to_numpy(cls, tensor: TensorType) -> np.ndarray:
        pass

    @classmethod
    def _check_assertion(cls, test: Union[TensorType, List[TensorType]],
                         reference: Union[TensorType, List[TensorType]],
                         assert_fn: Callable[[np.ndarray, np.ndarray], bool]):
        if not isinstance(test, list):
            test = [test]
        if not isinstance(reference, list):
            reference = [reference]
        assert len(test) == len(reference)

        for x, y in zip(test, reference):
            x = cls._to_numpy(x)
            y = cls._to_numpy(y)
            assert_fn(x, y)

    @classmethod
    def check_equal(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                    rtol: float = 1e-1):
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol))

    @classmethod
    def check_not_equal(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                        rtol: float = 1e-4):
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_raises(AssertionError,
                                                                                    np.testing.assert_allclose,
                                                                                    x, y, rtol=rtol))

    @classmethod
    def check_less(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                   rtol=1e-4):
        cls.check_not_equal(test, reference, rtol=rtol)
        cls._check_assertion(test, reference, np.testing.assert_array_less)

    @classmethod
    def check_greater(cls, test: Union[TensorType, List[TensorType]], reference: Union[TensorType, List[TensorType]],
                      rtol=1e-4):
        cls.check_not_equal(test, reference, rtol=rtol)
        cls._check_assertion(test, reference, lambda x, y: np.testing.assert_raises(AssertionError,
                                                                                    np.testing.assert_array_less,
                                                                                    x, y))
