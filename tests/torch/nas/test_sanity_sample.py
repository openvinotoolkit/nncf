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

import importlib
from pathlib import Path
from typing import Dict

import pytest

from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.search.search import NSGA2SearchAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import (
    ProgressiveShrinkingController,
)
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from tests.cross_fw.shared.command import arg_list_from_arg_dict
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.sample_test_validator import SampleType
from tests.torch.sample_test_validator import SanitySampleValidator
from tests.torch.sample_test_validator import SanityTestCaseDescriptor


class NASSampleTestDescriptor(SanityTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.sample_type(SampleType.CLASSIFICATION_NAS)
        self.mock_dataset("mock_32x32")
        self.batch_size(2)

    @property
    def config_directory(self) -> Path:
        return TEST_ROOT / "torch" / "data" / "configs" / "nas"

    def get_validator(self) -> "NASSampleValidator":
        if self.sample_type_ == SampleType.CLASSIFICATION_NAS_SEARCH:
            return NASSearchSampleValidator(self)
        return NASTrainSampleValidator(self)

    def get_compression_section(self):
        pass

    def get_config_update(self) -> Dict:
        sample_params = self.get_sample_params()
        sample_params["num_mock_images"] = 2
        sample_params["epochs"] = 5
        return sample_params


class NASSampleValidator(SanitySampleValidator):
    def __init__(self, desc: NASSampleTestDescriptor):
        super().__init__(desc)
        self._desc = desc
        self._all_spies = []

    def validate_spy(self):
        for spy in self._all_spies:
            spy.assert_called()


class NASTrainSampleValidator(NASSampleValidator):
    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        self._all_spies = [
            mocker.spy(ElasticWidthHandler, "get_random_config"),
            mocker.spy(ElasticWidthHandler, "reorganize_weights"),
            mocker.spy(ElasticDepthHandler, "get_random_config"),
            mocker.spy(MultiElasticityHandler, "activate_random_subnet"),
            mocker.spy(MultiElasticityHandler, "activate_minimum_subnet"),
            mocker.spy(MultiElasticityHandler, "activate_subnet_for_config"),
            mocker.spy(BootstrapNASScheduler, "epoch_step"),
            mocker.spy(BootstrapNASScheduler, "step"),
            mocker.spy(ProgressiveShrinkingController, "step"),
            mocker.spy(ProgressiveShrinkingController, "_run_batchnorm_adaptation"),
            mocker.spy(BatchnormAdaptationAlgorithm, "run"),
        ]


class NASSearchSampleValidator(NASSampleValidator):
    def setup_spy(self, mocker):
        super().setup_spy(mocker)
        self._all_spies = [
            mocker.spy(ElasticityController, "load_state"),
            mocker.spy(BaseSearchAlgorithm, "from_config"),
            mocker.spy(BaseSearchAlgorithm, "search_progression_to_csv"),
            mocker.spy(NSGA2SearchAlgorithm, "run"),
            mocker.spy(NSGA2SearchAlgorithm, "visualize_search_progression"),
            mocker.spy(NSGA2SearchAlgorithm, "evaluators_to_csv"),
            mocker.spy(MultiElasticityHandler, "activate_subnet_for_config"),
        ]


NAS_TEST_CASE_DESCRIPTORS = [
    NASSampleTestDescriptor().config_name("resnet50_cifar10_nas.json"),
    NASSampleTestDescriptor().config_name("mobilenet_v2_cifar10_nas.json"),
    NASSampleTestDescriptor().config_name("efficient_net_b1_cifar10_nas.json"),
    NASSampleTestDescriptor().config_name("mobilenet_v3_cifar10_nas.json"),
]


@pytest.fixture(name="nas_desc", params=NAS_TEST_CASE_DESCRIPTORS, ids=map(str, NAS_TEST_CASE_DESCRIPTORS))
def fixture_nas_desc(request, dataset_dir):
    desc: NASSampleTestDescriptor = request.param
    return desc.finalize(dataset_dir)


@pytest.mark.nightly
def test_e2e_supernet_training(nas_desc: NASSampleTestDescriptor, tmp_path, mocker):
    validator = nas_desc.get_validator()
    args = validator.get_default_args(tmp_path)
    validator.validate_sample(args, mocker)


@pytest.mark.nightly
def test_e2e_supernet_search(nas_desc: NASSampleTestDescriptor, tmp_path, mocker):
    # Train a supernet and save it to the specified path
    validator = nas_desc.get_validator()
    args = validator.get_default_args(tmp_path)
    output_dir = tmp_path / "output_dir"
    args["--checkpoint-save-dir"] = output_dir
    arg_list = arg_list_from_arg_dict(args)
    main_location = nas_desc.sample_handler.get_main_location()
    sample = importlib.import_module(main_location)
    sample.main(arg_list)

    nas_desc.sample_type(SampleType.CLASSIFICATION_NAS_SEARCH)
    validator = nas_desc.get_validator()
    args = validator.get_default_args(tmp_path)
    # Load the supernet to search
    args["--supernet-weights"] = output_dir / "last_model_weights.pth"
    args["--elasticity-state-path"] = output_dir / "last_elasticity.pth"
    args["--search-mode"] = True
    validator.validate_sample(args, mocker)
