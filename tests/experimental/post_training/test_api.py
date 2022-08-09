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

# pylint:disable=redefined-outer-name

import pytest

import numpy as np

from nncf.experimental.post_training.api.metric import Accuracy


@pytest.fixture
def outputs():
    return np.array(
        [
            [0.8, 0.1, 0.1],  # argmax = 0
            [0.1, 0.8, 0.1],  # argmax = 1
            [0.1, 0.1, 0.8],  # argmax = 2
            [0.8, 0.1, 0.1],  # argmax = 0
        ]
    )


@pytest.fixture
def acc_100(outputs):
    targets = np.array([0, 1, 2, 0])
    return outputs, targets


@pytest.fixture
def acc_0(outputs):
    targets = np.array([1, 2, 0, 1])
    return outputs, targets


class TestAccuracy:
    def test_acc_100(self, acc_100):
        metric = Accuracy()
        metric.update(*acc_100)

        assert metric.avg_value[metric.name] == 1.0

    def test_acc_0(self, acc_0):
        metric = Accuracy()
        metric.update(*acc_0)

        assert metric.avg_value[metric.name] == 0.0

    def test_acc_50(self, acc_100, acc_0):
        metric = Accuracy()

        metric.update(*acc_100)
        metric.update(*acc_0)

        assert metric.avg_value[metric.name] == 0.5
