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

from nncf.common.batchnorm_adaptation import BatchnormAdaptationAlgorithmImpl


class TFBatchnormAdaptationAlgorithmImpl(BatchnormAdaptationAlgorithmImpl):
    """
    Implementation of the batch-norm statistics adaptation algorithm for the TensorFlow backend.
    """

    def run(self, model) -> None:
        """
        Runs the batch-norm statistics adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """
        # TODO(andrey-churkin): Should be implemented.

        raise NotImplementedError('There is no possibility to start the batch-norm statistics adaptation algorithm '
                                  'for the TensorFlow backend because it is not implemented.')
