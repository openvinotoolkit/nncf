"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Dict

import pytest

from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import \
    ProgressiveShrinkingController
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from tests.common.helpers import TEST_ROOT
from tests.torch.test_sanity_sample import SanityTestCaseDescriptor
from tests.torch.test_sanity_sample import SampleType
from tests.torch.test_sanity_sample import get_default_args
from tests.torch.test_sanity_sample import validate_sample


class NASSampleTestDescriptor(SanityTestCaseDescriptor):
    def __init__(self):
        super().__init__()
        self.sample(SampleType.CLASSIFICATION)
        self.mock_dataset('mock_32x32')
        self.batch(2)
        self._all_spies = []

    def get_compression_section(self):
        pass

    def get_main_location(self):
        return 'examples.experimental.torch.classification.nas_advanced_main'

    def get_config_update(self) -> Dict:
        sample_params = self.get_sample_params()
        sample_params['num_mock_images'] = 2
        sample_params['epochs'] = 5
        return sample_params

    def setup_spy(self, mocker):
        # Need to mock SafeMLFLow to prevent starting a not closed mlflow session due to memory leak of config and
        # SafeMLFLow, which happens with a mocked train function
        mlflow_location = self.get_main_location() + '.SafeMLFLow'
        mocker.patch(mlflow_location)

        self._all_spies = [
            mocker.spy(ElasticWidthHandler, 'get_random_config'),
            mocker.spy(ElasticWidthHandler, 'reorganize_weights'),
            mocker.spy(ElasticDepthHandler, 'get_random_config'),
            mocker.spy(MultiElasticityHandler, 'activate_random_subnet'),
            mocker.spy(MultiElasticityHandler, 'activate_minimum_subnet'),
            mocker.spy(MultiElasticityHandler, 'set_config'),
            mocker.spy(BootstrapNASScheduler, 'epoch_step'),
            mocker.spy(BootstrapNASScheduler, 'step'),
            mocker.spy(ProgressiveShrinkingController, 'step'),
            mocker.spy(ProgressiveShrinkingController, '_run_batchnorm_adaptation'),
            mocker.spy(BatchnormAdaptationAlgorithm, 'run'),
        ]

    def validate_spy(self):
        for spy in self._all_spies:
            spy.assert_called()

    def dummy_config(self, config_name):
        self.config_name = config_name
        self.config_path = TEST_ROOT.joinpath("torch", "data", "configs", "nas") / config_name
        return self


NAS_TEST_CASE_DESCRIPTORS = [
    NASSampleTestDescriptor().dummy_config('resnet50_cifar10_nas.json'),
    NASSampleTestDescriptor().dummy_config('mobilenet_v2_cifar10_nas.json'),
    NASSampleTestDescriptor().dummy_config('efficient_net_b1_cifar10_nas.json')
]


@pytest.fixture(name='nas_desc', params=NAS_TEST_CASE_DESCRIPTORS, ids=map(str, NAS_TEST_CASE_DESCRIPTORS))
def fixture_nas_desc(request, dataset_dir):
    desc: NASSampleTestDescriptor = request.param
    return desc.finalize(dataset_dir)


def test_e2e_supernet_training(nas_desc: NASSampleTestDescriptor, tmp_path, mocker):
    args = get_default_args(nas_desc, tmp_path)
    validate_sample(args, nas_desc, mocker)
