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

from nncf.common.batchnorm_adaptation import BatchnormAdaptationAlgorithmImpl
from nncf.common.utils.logger import logger as nncf_logger
from nncf.initialization import DataLoaderBNAdaptationRunner


class PTBatchnormAdaptationAlgorithmImpl(BatchnormAdaptationAlgorithmImpl):
    """
    Implementation of the batch-norm adaptation algorithm for the PyTorch.
    """

    def run(self, model):
        """
        Runs the batch-norm adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """

        if self._extra_args is None:
            nncf_logger.info(
                'Could not run batchnorm adaptation '
                'as the adaptation data loader is not provided as an extra struct. '
                'Refer to `NNCFConfig.register_extra_structs` and the `BNAdaptationInitArgs` class')
            return

        batch_size = self._extra_args.data_loader.batch_size
        num_bn_forget_steps = np.ceil(self._num_bn_forget_samples / batch_size)
        num_bn_adaptation_steps = np.ceil(self._num_bn_adaptation_samples / batch_size)
        bn_adaptation_runner = DataLoaderBNAdaptationRunner(model, self._extra_args.device,
                                                            num_bn_forget_steps)
        bn_adaptation_runner.run(self._extra_args.data_loader, num_bn_adaptation_steps)
