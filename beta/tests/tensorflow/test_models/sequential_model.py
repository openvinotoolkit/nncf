"""
 Copyright (c) 2020 Intel Corporation
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

from tensorflow.keras import Sequential, layers


def SequentialModel(**kwargs):
    return Sequential([
        layers.InputLayer(kwargs['input_shape']),
        layers.Conv2D(10, 3),
        layers.MaxPool2D((4, 4)),
        layers.Flatten(),
        layers.Dense(20),
        layers.Dense(kwargs.get('classes', 10))
    ])

def SequentialModelNoInput(**kwargs):
    return Sequential([
        layers.Conv2D(10, 3),
        layers.MaxPool2D((4, 4)),
        layers.Flatten(),
        layers.Dense(20),
        layers.Dense(kwargs.get('classes', 10))
    ])
