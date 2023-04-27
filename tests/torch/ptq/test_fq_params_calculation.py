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

from typing import Any, Dict

import numpy as np
import pytest
import torch

import nncf
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.utils import get_all_modules_by_type
from tests.shared.helpers import compare_stats
from tests.shared.helpers import load_json
from tests.shared.paths import TEST_ROOT
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_random_mock_dataloader

REFERENCE_SCALES_DIR = TEST_ROOT / "torch" / "data" / "reference_scales"


def min_max_quantize_model(
    original_model: torch.nn.Module, quantization_params: Dict[str, Any] = None
) -> torch.nn.Module:
    config = nncf.NNCFConfig.from_dict({"input_info": {"sample_size": [1, 1, 10, 10]}})

    dataloader = create_random_mock_dataloader(config)

    def transform_fn(sample):
        inp, _ = sample
        return inp

    dataset = nncf.Dataset(dataloader, transform_func=transform_fn)

    post_training_quantization = PostTrainingQuantization(subset_size=1, **quantization_params)
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


def get_fq_nodes_params(model: NNCFNetwork) -> Dict[str, np.ndarray]:
    output = {}
    quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
    nncf_module_quantizations = get_all_modules_by_type(model, quantization_types)
    # This is not general logic, just used for the particular test
    nncf_module_quantizations.update(
        {str(scope): module[0] for scope, module in model.nncf.get_tracing_context()._post_hooks.items()}
    )

    for name, nncf_module_quantization in nncf_module_quantizations.items():
        input_low, input_high = nncf_module_quantization.get_input_low_input_high()
        input_low = input_low.cpu().detach().numpy()
        input_high = input_high.cpu().detach().numpy()
        output[str(name)] = {"input_low": input_low, "input_high": input_high}

    return output


@pytest.mark.parametrize(
    "overflow_fix",
    [OverflowFix.DISABLE, OverflowFix.ENABLE, OverflowFix.FIRST_LAYER],
    ids=[OverflowFix.DISABLE.value, OverflowFix.ENABLE.value, OverflowFix.FIRST_LAYER.value],
)
def test_overflow_fix_scales(_seed, overflow_fix):
    model = TwoConvTestModel()
    quantized_model = min_max_quantize_model(
        model, quantization_params={"advanced_parameters": AdvancedQuantizationParameters(overflow_fix=overflow_fix)}
    )
    fq_nodes_params = get_fq_nodes_params(quantized_model)

    ref_stats_name = "TwoConvTestModel" + f"_overflow_fix_{overflow_fix.value}.json"
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name

    # Unkomment lines below to generate reference for new models.
    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, fq_nodes_params)

    ref_nodes_params = load_json(ref_stats_path)
    params = ["input_low", "input_high"]
    compare_stats(ref_nodes_params, fq_nodes_params, params)
