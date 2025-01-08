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
from nncf.quantization.algorithms.weight_compression.mixed_precision import HAWQCriterion
from nncf.quantization.algorithms.weight_compression.mixed_precision import MaxVarianceCriterion
from nncf.quantization.algorithms.weight_compression.mixed_precision import MeanMaxCriterion
from nncf.quantization.algorithms.weight_compression.mixed_precision import MeanVarianceCriterion
from nncf.quantization.algorithms.weight_compression.openvino_backend import OVMixedPrecisionAlgoBackend
from tests.cross_fw.test_templates.test_weights_compression_backends import TemplateTestMixedPrecisionAlgoBackend
from tests.openvino.native.models import IdentityMatmul


class TestOVMixedPrecisionAlgoBackend(TemplateTestMixedPrecisionAlgoBackend):
    def get_hawq_with_backend(self, subset_size):
        hawq = HAWQCriterion(None, None, subset_size=subset_size)
        hawq._backend_entity = OVMixedPrecisionAlgoBackend(IdentityMatmul().ov_model)
        return hawq

    def get_mean_variance_with_backend(self, subset_size: int):
        mean_variance = MeanVarianceCriterion(None, None, subset_size=subset_size)
        mean_variance._backend_entity = OVMixedPrecisionAlgoBackend(IdentityMatmul().ov_model)
        return mean_variance

    def get_max_variance_with_backend(self, subset_size: int):
        max_variance = MaxVarianceCriterion(None, None, subset_size=subset_size)
        max_variance._backend_entity = OVMixedPrecisionAlgoBackend(IdentityMatmul().ov_model)
        return max_variance

    def get_mean_max_with_backend(self, subset_size: int):
        mean_max_variance = MeanMaxCriterion(None, None, subset_size=subset_size)
        mean_max_variance._backend_entity = OVMixedPrecisionAlgoBackend(IdentityMatmul().ov_model)
        return mean_max_variance
