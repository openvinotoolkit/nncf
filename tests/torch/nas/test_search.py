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

from typing import Any, Dict, List, NamedTuple

import pytest
import torch

from nncf import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.search.search import DataLoaderType
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import get_empty_config
from tests.torch.nas.creators import NAS_MODEL_DESCS
from tests.torch.nas.creators import create_bnas_model_and_ctrl_by_test_desc
from tests.torch.nas.creators import create_bootstrap_training_model_and_ctrl
from tests.torch.nas.models.synthetic import ThreeConvModel
from tests.torch.nas.models.synthetic import TwoConvAddConvTestModel
from tests.torch.nas.test_all_elasticity import fixture_nas_model_name  # noqa: F401

SEARCH_ALGORITHMS = ["NSGA2", "RNSGA2"]


@pytest.fixture(name="search_algo_name", scope="function", params=SEARCH_ALGORITHMS)
def fixture_search_algo_name(request):
    return request.param


class SearchTestDesc(NamedTuple):
    model_creator: Any
    # ref_model_stats: RefModelStats = None
    blocks_to_skip: List[List[str]] = None
    input_sizes: List[int] = [1, 3, 32, 32]
    algo_params: Dict = {}
    name: str = None
    mode: str = "auto"

    def __str__(self):
        if hasattr(self.model_creator, "__name__"):
            name = self.model_creator.__name__
        elif self.name is not None:
            name = self.name
        else:
            name = "NOT_DEFINED"
        return name


def prepare_test_model(search_desc, search_algo_name):
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(search_desc)
    elasticity_ctrl = ctrl.elasticity_controller
    config = {
        "input_info": {"sample_size": search_desc.input_sizes},
        "bootstrapNAS": {
            "training": {"elasticity": {"available_elasticity_dims": ["depth", "width"]}},
            "search": {
                "algorithm": search_algo_name,
                "num_evals": 2,
                "population": 1,
                "batchnorm_adaptation": {"num_bn_adaptation_samples": 2},
            },
        },
    }
    nncf_config = NNCFConfig.from_dict(config)
    bn_adapt_args = BNAdaptationInitArgs(data_loader=create_ones_mock_dataloader(nncf_config))
    nncf_config.register_extra_structs([bn_adapt_args])
    return model, elasticity_ctrl, nncf_config


def prepare_search_algorithm(nas_model_name, search_algo_name):
    if "inception_v3" in nas_model_name:
        pytest.skip(
            f"Skip test for {nas_model_name} as it fails because of 2 issues: "
            "not able to set DynamicInputOp to train-only layers (ticket 60976) and "
            "invalid padding update in elastic kernel (ticket 60990)"
        )

    elif nas_model_name in ["efficient_net_b0", "shufflenetv2"]:
        pytest.skip(f"Skip test for {nas_model_name} as exploration is underway to better manage its search space")
    model = NAS_MODEL_DESCS[nas_model_name][0]()
    nncf_config = get_empty_config(input_sample_sizes=NAS_MODEL_DESCS[nas_model_name][1])
    nncf_config["bootstrapNAS"] = {"training": {"algorithm": "progressive_shrinking"}}
    nncf_config["input_info"][0].update({"filler": "random"})
    nncf_config["bootstrapNAS"]["search"] = {"algorithm": search_algo_name, "num_evals": 2, "population": 1}
    nncf_config = NNCFConfig.from_dict(nncf_config)
    model, ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    elasticity_ctrl = ctrl.elasticity_controller
    elasticity_ctrl.multi_elasticity_handler.enable_all()
    return BaseSearchAlgorithm.from_config(model, elasticity_ctrl, nncf_config)


def update_search_bn_adapt_section(nncf_config, bn_adapt_section_is_called):
    if not bn_adapt_section_is_called:
        nncf_config["bootstrapNAS"]["search"]["batchnorm_adaptation"]["num_bn_adaptation_samples"] = 0


NAS_MODELS_SEARCH_ENCODING = {
    'resnet18': [1, 3, 7, 15, 1, 1, 3, 3, 7, 7, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 15],
    'resnet50': [7, 15, 31, 63, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7,
                 7, 7, 7, 7, 15, 15, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'densenet_121': [1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
                     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 7, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3,
                     0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0,
                     3, 0, 3, 0, 3, 0, 15, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0,
                     3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'mobilenet_v2': [0, 0, 1, 2, 4, 0, 0, 2, 3, 3, 5, 5, 5, 11, 11, 11, 11, 17, 17, 17, 29, 29, 29, 9,
                     39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31],
    'vgg11': [1, 3, 7, 7, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'vgg11_k7': [1, 3, 7, 7, 15, 15, 15, 15, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    'unet': [1, 3, 7, 15, 31, 15, 7, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0],
    'squeezenet1_0': [2, 0, 1, 1, 0, 1, 1, 0, 3, 3, 0, 3, 3, 0, 5, 5, 0, 5, 5, 1, 7, 7, 1, 7, 7, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'resnext29_32x4d': [7, 15, 31, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    'pnasnetb': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 0, 2, 0, 1, 0, 2, 0, 1, 0,
                 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1,
                 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0,
                 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 7],
    'ssd_mobilenet': [0, 1, 3, 3, 7, 7, 15, 15, 15, 15, 15, 15, 31, 31, 7, 15, 3, 7, 3, 7, 1, 3, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
    'ssd_vgg': [1, 1, 3, 3, 7, 7, 7, 15, 15, 15, 15, 15, 31, 31, 7, 15, 3, 7, 3, 7, 3, 7, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 7],
    'mobilenet_v3_small': [0, 0, 2, 6, 6, 0, 2, 3, 0, 8, 17, 17, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 3, 3,
                           17, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                           0, 0, 0, 255],
    'tcn': [3]
}  # fmt:skip


class TestSearchAlgorithm:
    def test_activate_maximum_subnet_at_init(self, search_algo_name):
        search_desc = SearchTestDesc(
            model_creator=ThreeConvModel,
            algo_params={"width": {"min_width": 1, "width_step": 1}},
            input_sizes=ThreeConvModel.INPUT_SIZE,
        )
        model, elasticity_ctrl, nncf_config = prepare_test_model(search_desc, search_algo_name)
        elasticity_ctrl.multi_elasticity_handler.enable_elasticity(ElasticityDim.WIDTH)
        BaseSearchAlgorithm.from_config(model, elasticity_ctrl, nncf_config)
        config_init = elasticity_ctrl.multi_elasticity_handler.get_active_config()
        elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        assert config_init == elasticity_ctrl.multi_elasticity_handler.get_active_config()

    def test_design_upper_bounds(self, nas_model_name, search_algo_name):
        search = prepare_search_algorithm(nas_model_name, search_algo_name)
        assert search.vars_upper == NAS_MODELS_SEARCH_ENCODING[nas_model_name]
        assert search.num_vars == len(NAS_MODELS_SEARCH_ENCODING[nas_model_name])

    @pytest.mark.parametrize(
        "bn_adapt_section_is_called",
        [False, True],
        ids=["section_with_zero_num_samples", "section_with_non_zero_num_samples"],
    )
    def test_bn_adapt(self, mocker, bn_adapt_section_is_called, tmp_path, search_algo_name):
        search_desc = SearchTestDesc(
            model_creator=ThreeConvModel,
            algo_params={"width": {"min_width": 1, "width_step": 1}},
            input_sizes=ThreeConvModel.INPUT_SIZE,
        )
        nncf_network, ctrl, nncf_config = prepare_test_model(search_desc, search_algo_name)
        update_search_bn_adapt_section(nncf_config, bn_adapt_section_is_called)
        bn_adapt_run_patch = mocker.patch(
            "nncf.common.initialization.batchnorm_adaptation.BatchnormAdaptationAlgorithm.run"
        )
        ctrl.multi_elasticity_handler.enable_all()
        search_algo = BaseSearchAlgorithm.from_config(nncf_network, ctrl, nncf_config)

        def fake_acc_eval(*unused):
            return 0

        search_algo.run(fake_acc_eval, mocker.MagicMock(spec=DataLoaderType), tmp_path)
        if bn_adapt_section_is_called:
            bn_adapt_run_patch.assert_called()
        else:
            bn_adapt_run_patch.assert_not_called()


class SearchTestResultDesc(NamedTuple):
    model_creator: Any
    expected_accuracy: float
    subnet_expected_accuracy: Dict
    input_sizes: List[int]
    search_spaces: Dict
    eval_datasets: List


SEARCH_RESULT_DESCRIPTORS = [
    SearchTestResultDesc(
        model_creator=TwoConvAddConvTestModel,
        expected_accuracy=0.238,
        subnet_expected_accuracy={
            SEARCH_ALGORITHMS[0]: 0.476,
            SEARCH_ALGORITHMS[1]: 0.476,
        },
        input_sizes=TwoConvAddConvTestModel.INPUT_SIZE,
        search_spaces={
            (
                "TwoConvAddConvTestModel/NNCFConv2d[conv1]/conv2d_0",
                "TwoConvAddConvTestModel/NNCFConv2d[conv2]/conv2d_0",
            ): [3, 2, 1],
        },
        eval_datasets=[
            (torch.Tensor([item / 10.0]).reshape(TwoConvAddConvTestModel.INPUT_SIZE), int(item > 0))
            for item in range(-10, 11)
        ],  # (input, label)
    )
]


@pytest.mark.parametrize(
    "search_result_descriptors", SEARCH_RESULT_DESCRIPTORS, ids=map(str, SEARCH_RESULT_DESCRIPTORS)
)
def test_search_results(search_result_descriptors, search_algo_name, tmp_path):
    config = {
        "input_info": {"sample_size": search_result_descriptors.input_sizes},
        "bootstrapNAS": {
            "training": {
                "elasticity": {
                    "available_elasticity_dims": ["width"],
                    "width": {"overwrite_groups": [], "overwrite_groups_widths": []},
                }
            },
            "search": {"algorithm": search_algo_name, "num_evals": 10, "population": 2},
        },
    }
    for group, width in search_result_descriptors.search_spaces.items():
        config["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"].append(list(group))
        config["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups_widths"].append(width)
    nncf_config = NNCFConfig.from_dict(config)
    model, ctrl = create_bootstrap_training_model_and_ctrl(search_result_descriptors.model_creator(), nncf_config)
    model.eval()
    device = next(model.parameters()).device
    elasticity_ctrl = ctrl.elasticity_controller
    elasticity_ctrl.multi_elasticity_handler.enable_all()
    search = BaseSearchAlgorithm.from_config(model, elasticity_ctrl, nncf_config)

    # (input, label)
    eval_datasets = [(item[0].to(device), item[1]) for item in search_result_descriptors.eval_datasets]

    def validate_model_fn(model, eval_datasets):
        count = 0
        with torch.no_grad():
            for input, label in eval_datasets:
                output = model(input)
                pred = int((output < 40.0).item())  # binary classification
                count += pred == label
        return round(count / len(eval_datasets), 3)

    elasticity_ctrl.multi_elasticity_handler.activate_supernet()
    max_subnetwork_macs = (
        elasticity_ctrl.multi_elasticity_handler.count_flops_and_weights_for_active_subnet()[0] / 2000000
    )
    max_subnetwork_acc = validate_model_fn(model, eval_datasets)

    _, _, performance_metrics = search.run(validate_model_fn, eval_datasets, tmp_path)

    assert max_subnetwork_acc == search_result_descriptors.expected_accuracy
    assert performance_metrics[1] == search_result_descriptors.subnet_expected_accuracy[search_algo_name]
    assert performance_metrics[0] < max_subnetwork_macs
    assert performance_metrics[1] > max_subnetwork_acc


class TestSearchEvaluators:
    def test_create_default_evaluators(self, nas_model_name, search_algo_name, tmp_path):
        if nas_model_name in ["squeezenet1_0", "pnasnetb"]:
            pytest.skip(f"Skip test for {nas_model_name} as it fails.")
        search = prepare_search_algorithm(nas_model_name, search_algo_name)
        search.run(lambda model, val_loader: 0, None, tmp_path)
        evaluators = search.evaluator_handlers
        assert len(evaluators) == 2
        assert evaluators[0].name == "MACs"
        assert evaluators[1].name == "top1_acc"
