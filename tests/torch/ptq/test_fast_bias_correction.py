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
import pytest

from nncf.data import Dataset
from nncf.torch.nncf_network import ExtraCompressionModuleType
from tests.torch import test_models
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import RandomDatasetMock
from tests.torch.ptq.helpers import get_min_max_and_fast_bias_correction_algo_for_test
from tests.torch.ptq.helpers import get_nncf_network
from tests.torch.test_compressed_graph import ModelDesc


def get_model_name(desc):
    if isinstance(desc, ModelDesc):
        return desc.model_name
    return desc.values[0].model_name


TEST_MODELS_DESC = [
    ModelDesc("resnet18", test_models.ResNet18, [1, 3, 32, 32]),
    ModelDesc("BasicConvTestModel", BasicConvTestModel, [1, 1, 4, 4])
]

@pytest.mark.parametrize("desc", TEST_MODELS_DESC, ids=[get_model_name(m) for m in TEST_MODELS_DESC])
def test_fast_bias_correction_algo(desc: ModelDesc):
    model = desc.model_builder()

    nncf_network = get_nncf_network(model, desc.input_sample_sizes)
    nncf_network.register_compression_module_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER)
    quantization_algorithm = get_min_max_and_fast_bias_correction_algo_for_test()

    def transform_fn(data_item):
        images, _ = data_item
        return images

    dataset = Dataset(RandomDatasetMock(desc.input_sample_sizes), transform_fn)
    quantized_model = quantization_algorithm.apply(nncf_network, dataset=dataset)
