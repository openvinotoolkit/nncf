"""
 Copyright (c) 2023 Intel Corporation
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

from nncf.tensorflow.loss import TFZeroCompressionLoss

# pylint: disable=use-implicit-booleaness-not-comparison
def test_tf_zero_loss_state():
    loss = TFZeroCompressionLoss()
    assert loss.get_state() == {}
    loss.load_state({})
    assert loss.get_state() == {}
