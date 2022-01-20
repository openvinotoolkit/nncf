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

from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.tensorflow.pruning.base_algorithm import BasePruningAlgoController
from nncf.tensorflow.pruning.callbacks import PruningStatisticsCallback
from nncf.tensorflow.sparsity.callbacks import SparsityStatisticsCallback
from nncf.tensorflow.sparsity.callbacks import UpdateMask
from nncf.tensorflow.sparsity.base_algorithm import BaseSparsityController


def create_compression_callbacks(compression_ctrl, log_tensorboard=True, log_text=True, log_dir=None):
    compression_controllers = compression_ctrl.child_ctrls \
        if isinstance(compression_ctrl, CompositeCompressionAlgorithmController) \
        else [compression_ctrl]
    for ctrl in compression_controllers:
        if isinstance(ctrl, (BaseSparsityController, BasePruningAlgoController)):
            callbacks = [UpdateMask(ctrl.scheduler)]
            if log_tensorboard or log_text:
                if isinstance(ctrl, BaseSparsityController):
                    statistics_callback_cls = SparsityStatisticsCallback
                else:
                    statistics_callback_cls = PruningStatisticsCallback

                callbacks += [statistics_callback_cls(ctrl.statistics,
                                                      log_tensorboard=log_tensorboard,
                                                      log_text=log_text,
                                                      log_dir=log_dir)]
            return callbacks
    return []
