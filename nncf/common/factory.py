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

from typing import Optional, List, Tuple, Any

from nncf.api.compression import ModelType
from nncf.common.exporter import Exporter
from nncf.config.structure import BNAdaptationInitArgs

from nncf.common.utils.backend import __nncf_backend__
if __nncf_backend__ == 'Torch':
    from nncf.torch.exporter import PTExporter
    from nncf.torch.batchnorm_adaptation import PTBatchnormAdaptationAlgorithmImpl
elif __nncf_backend__ == 'TensorFlow':
    from beta.nncf.tensorflow.exporter import TFExporter
    from beta.nncf.tensorflow.batchnorm_adaptation import TFBatchnormAdaptationAlgorithmImpl


def create_exporter(model: ModelType,
                    input_names: Optional[List[str]] = None,
                    output_names: Optional[List[str]] = None,
                    model_args: Optional[Tuple[Any, ...]] = None) -> Exporter:
    """
    Factory for building an exporter.
    """
    if __nncf_backend__ == 'Torch':
        exporter = PTExporter(model, input_names, output_names, model_args)
    elif __nncf_backend__ == 'TensorFlow':
        exporter = TFExporter(model, input_names, output_names, model_args)

    return exporter


def create_bn_adaptation_algorithm_impl(num_bn_adaptation_samples: int,
                                        num_bn_forget_samples: int,
                                        extra_args: BNAdaptationInitArgs):
    """
    Factory for building a batchnorm adaptation algorithm implementation.
    """
    if __nncf_backend__ == 'Torch':
        bn_adaptation_algorithm_impl = PTBatchnormAdaptationAlgorithmImpl(num_bn_adaptation_samples,
                                                                          num_bn_forget_samples,
                                                                          extra_args)
    elif __nncf_backend__ == 'Tensorflow':
        bn_adaptation_algorithm_impl = TFBatchnormAdaptationAlgorithmImpl(num_bn_adaptation_samples,
                                                                          num_bn_forget_samples,
                                                                          extra_args)

    return bn_adaptation_algorithm_impl
