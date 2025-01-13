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

from functools import partial
from pathlib import Path

import pytest
import torch

from nncf import Dataset
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.torch import wrap_model
from nncf.torch.layers import NNCF_RNN
from nncf.torch.layers import LSTMCellNNCF
from tests.cross_fw.test_templates.helpers import EmbeddingModel
from tests.cross_fw.test_templates.helpers import RoPEModel
from tests.cross_fw.test_templates.helpers import ScaledDotProductAttentionModel
from tests.torch import test_models
from tests.torch.quantization.test_algo_quantization import SharedLayersModel
from tests.torch.test_compressed_graph import ModelDesc
from tests.torch.test_compressed_graph import check_graph

# Use the same graphs for min_max quantization as for symmetric quantization
ALGOS = ["symmetric"]
SKIP_MARK = pytest.mark.skip("Model is not supported yet")


@pytest.fixture(scope="function", params=ALGOS, name="graph_dir")
def fixture_graph_dir(request):
    quantization_type = request.param
    graph_dir_ = Path("quantized") / "ptq" / quantization_type
    return graph_dir_


def get_model_name(description):
    desc = description[0]
    if isinstance(desc, ModelDesc):
        return desc.model_name
    return desc[0].model_name


TEST_MODELS_DESC = [
    (ModelDesc("embedding_model", EmbeddingModel, [1, 10]), {}),
    (ModelDesc("rope_model", RoPEModel, [1, 10]), {"model_type": ModelType.TRANSFORMER}),
    (
        ModelDesc(
            "scaled_dot_product_attention_model",
            ScaledDotProductAttentionModel,
            {"query": [1, 8, 16], "key": [1, 8, 16], "value": [1, 8, 16]},
        ),
        {},
    ),
    (ModelDesc("shared_model", SharedLayersModel, [1, 1, 5, 6]), {}),
    (ModelDesc("alexnet", test_models.AlexNet, [1, 3, 32, 32]), {}),
    (ModelDesc("lenet", test_models.LeNet, [1, 3, 32, 32]), {}),
    (ModelDesc("resnet18", test_models.ResNet18, [1, 3, 32, 32]), {}),
    (ModelDesc("resnet50_cpu_spr", test_models.ResNet50, [1, 3, 32, 32]), {"target_device": TargetDevice.CPU_SPR}),
    (ModelDesc("vgg16", partial(test_models.VGG, "VGG16"), [1, 3, 32, 32]), {}),
    (ModelDesc("inception", test_models.GoogLeNet, [1, 3, 32, 32]), {}),
    (ModelDesc("densenet121", test_models.DenseNet121, [1, 3, 32, 32]), {}),
    (
        ModelDesc(
            "inception_v3", partial(test_models.Inception3, aux_logits=True, transform_input=True), [2, 3, 299, 299]
        ),
        {},
    ),
    (ModelDesc("squeezenet1_1", test_models.squeezenet1_1, [1, 3, 32, 32]), {}),
    (ModelDesc("shufflenetv2", partial(test_models.ShuffleNetV2, net_size=0.5), [1, 3, 32, 32]), {}),
    (ModelDesc("ssd_vgg", test_models.ssd_vgg300, [2, 3, 300, 300]), {}),
    (ModelDesc("ssd_mobilenet", test_models.ssd_mobilenet, [2, 3, 300, 300]), {}),
    (ModelDesc("mobilenet_v2", test_models.mobilenet_v2, [2, 3, 32, 32]), {}),
    (ModelDesc("mobilenet_v3_small", test_models.mobilenet_v3_small, [2, 3, 32, 32]), {}),
    (ModelDesc("unet", test_models.UNet, [1, 3, 360, 480]), {}),
    pytest.param(ModelDesc("lstm_cell", LSTMCellNNCF, [2, 1]), {}, marks=SKIP_MARK),
    pytest.param(
        ModelDesc("lstm_uni_seq", partial(NNCF_RNN, num_layers=1, bidirectional=False), [3, 1, 1]), {}, marks=SKIP_MARK
    ),
    pytest.param(
        ModelDesc("lstm_uni_stacked", partial(NNCF_RNN, num_layers=2, bidirectional=False), [3, 1, 1]),
        {},
        marks=SKIP_MARK,
    ),
    pytest.param(
        ModelDesc("lstm_bi_seq", partial(NNCF_RNN, num_layers=1, bidirectional=True), [3, 1, 1]), {}, marks=SKIP_MARK
    ),
    pytest.param(
        ModelDesc("lstm_bi_stacked", partial(NNCF_RNN, num_layers=2, bidirectional=True), [3, 1, 1]),
        {},
        marks=SKIP_MARK,
    ),
]


@pytest.mark.parametrize(
    ("desc", "quantization_parameters"), TEST_MODELS_DESC, ids=[get_model_name(m) for m in TEST_MODELS_DESC]
)
def test_min_max_classification_quantized_graphs(desc: ModelDesc, quantization_parameters, graph_dir, mocker):
    model = desc.model_builder()

    if isinstance(desc.input_sample_sizes, dict):
        example_input = {}
        for name, size in desc.input_sample_sizes.items():
            example_input[name] = torch.ones(size)
    else:
        example_input = torch.ones(desc.input_sample_sizes)

    nncf_network = wrap_model(model, example_input, trace_parameters=True)
    quantization_parameters["advanced_parameters"] = AdvancedQuantizationParameters(disable_bias_correction=True)
    quantization_parameters["subset_size"] = 1
    quantization_algorithm = PostTrainingQuantization(**quantization_parameters)

    quantized_model = quantization_algorithm.apply(
        nncf_network,
        nncf_network.nncf.get_graph(),
        dataset=Dataset([example_input]),
    )
    check_graph(quantized_model.nncf.get_graph(), desc.dot_filename(), graph_dir)
