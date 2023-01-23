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

from pathlib import Path
from typing import Dict

import pytest
import torch
from pkg_resources import parse_version

from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import \
    ProgressiveShrinkingController
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from tests.shared.paths import TEST_ROOT
from tests.torch.sample_test_validator import SampleType
from tests.torch.sample_test_validator import SanitySampleValidator
from tests.torch.sample_test_validator import SanityTestCaseDescriptor


class NASSampleTestDescriptor(SanityTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.sample_type(SampleType.CLASSIFICATION_NAS)
        self.mock_dataset('mock_32x32')
        self.batch_size(2)

    @property
    def config_directory(self) -> Path:
        return TEST_ROOT / "torch" / "data" / "configs" / "nas"

    def get_validator(self) -> 'NASSampleValidator':
        return NASSampleValidator(self)

    def get_compression_section(self):
        pass

    def get_config_update(self) -> Dict:
        sample_params = self.get_sample_params()
        sample_params['num_mock_images'] = 2
        sample_params['epochs'] = 5
        return sample_params


class NASSampleValidator(SanitySampleValidator):
    def __init__(self, desc: NASSampleTestDescriptor):
        super().__init__(desc)
        self._desc = desc
        self._all_spies = []

    def setup_spy(self, mocker):
        # Need to mock SafeMLFLow to prevent starting a not closed mlflow session due to memory leak of config and
        # SafeMLFLow, which happens with a mocked train function
        self._sample_handler.mock_mlflow(mocker)

        self._all_spies = [
            mocker.spy(ElasticWidthHandler, 'get_random_config'),
            mocker.spy(ElasticWidthHandler, 'reorganize_weights'),
            mocker.spy(ElasticDepthHandler, 'get_random_config'),
            mocker.spy(MultiElasticityHandler, 'activate_random_subnet'),
            mocker.spy(MultiElasticityHandler, 'activate_minimum_subnet'),
            mocker.spy(MultiElasticityHandler, 'activate_subnet_for_config'),
            mocker.spy(BootstrapNASScheduler, 'epoch_step'),
            mocker.spy(BootstrapNASScheduler, 'step'),
            mocker.spy(ProgressiveShrinkingController, 'step'),
            mocker.spy(ProgressiveShrinkingController, '_run_batchnorm_adaptation'),
            mocker.spy(BatchnormAdaptationAlgorithm, 'run'),
        ]

    def validate_spy(self):
        for spy in self._all_spies:
            spy.assert_called()


NAS_TEST_CASE_DESCRIPTORS = [
    NASSampleTestDescriptor().config_name('resnet50_cifar10_nas.json'),
    NASSampleTestDescriptor().config_name('mobilenet_v2_cifar10_nas.json'),
    NASSampleTestDescriptor().config_name('efficient_net_b1_cifar10_nas.json'),
    NASSampleTestDescriptor().config_name('mobilenet_v3_cifar10_nas.json')
]


@pytest.fixture(name='nas_desc', params=NAS_TEST_CASE_DESCRIPTORS, ids=map(str, NAS_TEST_CASE_DESCRIPTORS))
def fixture_nas_desc(request, dataset_dir):
    desc: NASSampleTestDescriptor = request.param
    return desc.finalize(dataset_dir)


def test_e2e_supernet_training(nas_desc: NASSampleTestDescriptor, tmp_path, mocker):
    if parse_version(torch.__version__) < parse_version("1.9") and \
            ('efficient_net' in nas_desc.config_name_ or 'mobilenet_v3' in nas_desc.config_name_):
        pytest.skip(f'Test exports model with hardsigmoid operator to ONNX opset version 13.\n'
                    f'It is not supported in the current torch version: {torch.__version__}')
    validator = nas_desc.get_validator()
    args = validator.get_default_args(tmp_path)
    validator.validate_sample(args, mocker)
