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

from typing import Dict, Any

import pytest
import numpy as np
import json
from copy import deepcopy

import nncf
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.utils import get_all_modules_by_type
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.definitions import OverflowFix
from nncf.torch.model_creation import create_nncf_network

from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_random_mock_dataloader
from tests.shared.paths import TEST_ROOT

TORCH_TEST_ROOT = TEST_ROOT / 'torch'
REFERENCE_SCALES_DIR = TORCH_TEST_ROOT / 'data' / 'reference_scales'

def load_json(stats_path):
    with open(stats_path, 'r', encoding='utf8') as json_file:
        return json.load(json_file)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    # pylint: disable=W0221, E0202

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def dump_to_json(local_path, data):
    with open(local_path, 'w', encoding='utf8') as file:
        json.dump(deepcopy(data), file, indent=4, cls=NumpyEncoder)

def compare_stats(expected, actual):
    assert len(expected) == len(actual)
    for ref_name in expected:
        ref_stats = expected[ref_name]
        ref_input_low, ref_input_high = ref_stats['input_low'], ref_stats['input_high']

        stats = actual[ref_name]
        input_low, input_high = stats['input_low'], stats['input_high']

        assert np.allclose(ref_input_low, input_low, atol=1e-6)
        assert np.allclose(ref_input_high, input_high, atol=1e-6)


def min_max_quantize_model(original_model, quantization_params: Dict[str, Any] = None):
    config = nncf.NNCFConfig.from_dict({'input_info': 
                                            {'sample_size': [1, 1, 10, 10]}
        })
    
    dataset = create_random_mock_dataloader(config)
    def transform_fn(sample):
        inp, target = sample
        return inp
    dataset = nncf.Dataset(dataset, transform_func=transform_fn)
    
    post_training_quantization = PostTrainingQuantization(
        PostTrainingQuantizationParameters(number_samples=1, **quantization_params))
    # Using PTQ, but apply only MinMax
    updated_algorithms = []
    for algo in post_training_quantization.algorithms:
        if isinstance(algo, MinMaxQuantization):
            updated_algorithms.append(algo)
    post_training_quantization.algorithms = updated_algorithms
    
    original_model.eval()
    nncf_network = create_nncf_network(original_model, config)
    
    quantized_model = post_training_quantization.apply(nncf_network, dataset=dataset)
    
    return quantized_model


def get_fq_nodes(model):
    output = {}

    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    nncf_module_quantizations = get_all_modules_by_type(model, quantization_types)

    for quantization_type in quantization_types:
        nncf_module_quantizations.update(get_all_modules_by_type(model, quantization_type))
    for name, nncf_module_quantization in nncf_module_quantizations.items():
        input_low, input_high = nncf_module_quantization.get_input_low_input_high()
        output[str(name)] = {'input_low': input_low.cpu().detach().numpy(), 'input_high': input_high.cpu().detach().numpy()}
    
    return output

@pytest.mark.parametrize('overflow_fix', [OverflowFix.DISABLE, OverflowFix.ENABLE, OverflowFix.FIRST_LAYER],
                         ids=[OverflowFix.DISABLE.value, OverflowFix.ENABLE.value, OverflowFix.FIRST_LAYER.value])
def test_overflow_fix_scales(overflow_fix):
    model = TwoConvTestModel()
    quantized_model = min_max_quantize_model(model, quantization_params={'overflow_fix': overflow_fix})
    nodes = get_fq_nodes(quantized_model)

    ref_stats_name = 'TwoConvTestModel' + f'_overflow_fix_{overflow_fix.value}.json'
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name

    # Unkomment lines below to generate reference for new models.
    # dump_to_json(ref_stats_path, nodes)

    ref_nodes = load_json(ref_stats_path)
    compare_stats(ref_nodes, nodes)