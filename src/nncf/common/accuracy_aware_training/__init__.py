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
"""
Accuracy Aware Training functionality.
"""

from nncf.common.accuracy_aware_training.training_loop import AccuracyAwareTrainingMode as AccuracyAwareTrainingMode
from nncf.common.accuracy_aware_training.training_loop import (
    create_accuracy_aware_training_loop as create_accuracy_aware_training_loop,
)
