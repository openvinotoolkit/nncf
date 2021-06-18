"""
 Copyright (c) 2019-2020 Intel Corporation
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
import copy
import itertools
import os
from functools import partial
from functools import reduce
from typing import Dict
from typing import List

import onnx
import pytest
from torch import cuda
from torch import nn
from torch.nn import DataParallel

from nncf import NNCFConfig
from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS
from nncf.torch.algo_selector import NoCompressionAlgorithmBuilder
from nncf.api.compression import CompressionStage
from nncf.torch.checkpoint_loading import load_state
from nncf.common.hardware.config import HWConfigType
from nncf.torch.compression_method_api import DOMAIN_CUSTOM_OPS_NAME
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.torch.sparsity.rb.test_algo import get_basic_sparsity_config


class BasicLinearTestModel(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.fc = nn.Linear(size, size)

    def forward(self, x):
        return self.fc(x)

class BasicTestModelWithTwoInputOutput(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.fc0 = nn.Linear(size, size)
        self.fc1 = nn.Linear(size, size)

    def forward(self, x0, x1):
        return self.fc0(x0), self.fc1(x1)

def get_const_sparsity_config():
    config = get_empty_config()
    config['compression'] = {'algorithm': 'const_sparsity'}
    return config


def get_basic_asym_quantization_config(model_size=4):
    config = get_quantization_config_without_range_init(model_size)
    config['compression']['activations'] = {"mode": "asymmetric"}
    config['compression']['initializer']['range'] = {"num_init_samples": 0}
    return config


@pytest.mark.parametrize('config_provider',
                         (get_quantization_config_without_range_init, get_basic_asym_quantization_config,
                          get_basic_sparsity_config,
                          get_basic_magnitude_sparsity_config, get_const_sparsity_config),
                         ids=('SymQuantization', 'AsymQuantization', 'Sparsity', 'MagnitudeSparsity', 'ConstSparsity'))
@pytest.mark.parametrize('model_provider', (BasicConvTestModel, BasicLinearTestModel),
                         ids=('Conv2d', 'Linear'))
class TestCompressionAlgos:
    def test_can_export_compressed_model(self, tmp_path, config_provider, model_provider):
        test_path = str(tmp_path.joinpath('test.onnx'))
        model = model_provider()
        config = config_provider()
        register_bn_adaptation_init_args(config)
        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        compression_ctrl.export_model(test_path)
        assert os.path.exists(test_path)

class TestConfigCreator:
    def __init__(self):
        self._config = get_empty_config()
        self._algorithm_sections = {}

    def create(self) -> NNCFConfig:
        self._config['compression'] = []
        for algo_name, params in self._algorithm_sections.items():
            algo_section = {'algorithm': algo_name}
            if params:
                algo_section['params'] = params
            self._config['compression'].append(algo_section)
        return self._config

    def add_algo(self, name: str, params: Dict = None):
        self._algorithm_sections[name] = params
        return self

    def __str__(self):
        return '_'.join(self._algorithm_sections)


class CompressionStageTestStruct:
    def __init__(self, config_provider: 'TestConfigCreator', compression_stages: List[CompressionStage]):
        self.config_provider = config_provider
        self.compression_stages = compression_stages

    def __str__(self):
        return str(self.config_provider)


staged_quantization_params = {'activations_quant_start_epoch': 1, 'weights_quant_start_epoch': 2}
magnitude_sparsity_params = {'schedule': 'multistep',
                             'multistep_steps': [1, 2],
                             'multistep_sparsity_levels': [0, 0.3, 0.5]}
filter_pruning_params = {'schedule': 'exponential', 'num_init_steps': 0, 'pruning_steps': 3}
FFF_levels = [CompressionStage.FULLY_COMPRESSED] * 3
NPF_levels = [CompressionStage.UNCOMPRESSED, CompressionStage.PARTIALLY_COMPRESSED, CompressionStage.FULLY_COMPRESSED]
LIST_OF_TEST_PARAMS = [
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('quantization'),
        compression_stages=FFF_levels
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('quantization', staged_quantization_params),
        compression_stages=NPF_levels
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('const_sparsity'),
        compression_stages=FFF_levels
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('magnitude_sparsity', magnitude_sparsity_params),
        compression_stages=NPF_levels
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('rb_sparsity', {
            'sparsity_target': 0.61,
            'sparsity_target_epoch': 2,
        }),
        compression_stages=NPF_levels
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('filter_pruning', {
            'num_init_steps': 1,
            'pruning_steps': 2,
        }),
        compression_stages=[CompressionStage.UNCOMPRESSED,
                            CompressionStage.FULLY_COMPRESSED,
                            CompressionStage.FULLY_COMPRESSED]
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('filter_pruning', filter_pruning_params),
        compression_stages=NPF_levels
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('magnitude_sparsity', magnitude_sparsity_params).add_algo(
            'quantization'),
        compression_stages=[CompressionStage.PARTIALLY_COMPRESSED] * 2 + [CompressionStage.FULLY_COMPRESSED],
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('magnitude_sparsity', magnitude_sparsity_params).add_algo(
            'quantization', staged_quantization_params),
        compression_stages=NPF_levels,
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('quantization', staged_quantization_params).add_algo(
            'filter_pruning', filter_pruning_params),
        compression_stages=NPF_levels,
    ),
    CompressionStageTestStruct(
        config_provider=TestConfigCreator().add_algo('magnitude_sparsity', magnitude_sparsity_params).add_algo(
            'quantization', staged_quantization_params).add_algo('filter_pruning', filter_pruning_params),
        compression_stages=NPF_levels,
    ),
]


@pytest.mark.parametrize('test_struct', LIST_OF_TEST_PARAMS, ids=[str(param) for param in LIST_OF_TEST_PARAMS])
def test_can_get_compression_stage(test_struct: CompressionStageTestStruct):
    config_provider, compression_stages = test_struct.config_provider, test_struct.compression_stages
    model = BasicConvTestModel()
    config = config_provider.create()
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_scheduler = compression_ctrl.scheduler
    assert compression_ctrl.compression_stage() == compression_stages[0]

    compression_scheduler.epoch_step()
    assert compression_ctrl.compression_stage() == compression_stages[0]

    compression_scheduler.epoch_step()
    assert compression_ctrl.compression_stage() == compression_stages[1]

    compression_scheduler.epoch_step()
    assert compression_ctrl.compression_stage() == compression_stages[2]


@pytest.mark.parametrize(('src', 'dst', 'ref'),
                            (
                             (CompressionStage.UNCOMPRESSED,
                              CompressionStage.UNCOMPRESSED,
                              CompressionStage.UNCOMPRESSED),
                             (CompressionStage.PARTIALLY_COMPRESSED,
                              CompressionStage.PARTIALLY_COMPRESSED,
                              CompressionStage.PARTIALLY_COMPRESSED),
                             (CompressionStage.FULLY_COMPRESSED,
                              CompressionStage.FULLY_COMPRESSED,
                              CompressionStage.FULLY_COMPRESSED),
                             (CompressionStage.UNCOMPRESSED,
                              CompressionStage.PARTIALLY_COMPRESSED,
                              CompressionStage.PARTIALLY_COMPRESSED),
                             (CompressionStage.UNCOMPRESSED,
                              CompressionStage.FULLY_COMPRESSED,
                              CompressionStage.PARTIALLY_COMPRESSED),
                             (CompressionStage.PARTIALLY_COMPRESSED,
                              CompressionStage.FULLY_COMPRESSED,
                              CompressionStage.PARTIALLY_COMPRESSED)
                              )
                         )
def test_combo_of_compression_stages(src, dst, ref):
    assert src + dst == ref
    assert dst + src == ref
    src_c = copy.deepcopy(src)
    src_c += dst
    assert src_c == ref
    dst_c = copy.deepcopy(dst)
    dst_c += src
    assert dst_c == ref


QUANTIZATION = 'quantization'
SPARSITY_TYPES = ['magnitude', 'rb', 'const']
SPARSITY_ALGOS = ['_'.join([type, 'sparsity']) for type in SPARSITY_TYPES]  # 3S

LOAD_ALGOS = list(itertools.product([QUANTIZATION], SPARSITY_ALGOS))  # Q + 3S
LOAD_ALGOS += itertools.product(SPARSITY_ALGOS, [QUANTIZATION])  # 3S + Q

SAVE_ALGOS = [[algo] for algo in SPARSITY_ALGOS]  # 3S
SAVE_ALGOS += [[QUANTIZATION]]  # Q
SAVE_ALGOS += LOAD_ALGOS  # Q , 3S, 3S + Q, Q+3S

ALGOS = list(itertools.product(SAVE_ALGOS, LOAD_ALGOS))


@pytest.fixture(scope='module', params=ALGOS,
                ids=['__'.join(['save:' + '_'.join(a[0]),
                                'load:' + '_'.join(a[1])]) for a in ALGOS]
                )
def _algos(request):
    pair_algos = request.param
    save_algos = pair_algos[0]
    load_algos = pair_algos[1]
    resume_ok = False
    # resume expects the same list of algorithms
    if save_algos == load_algos:
        resume_ok = True

    if len(save_algos) == len(load_algos):
        for s, v in zip(save_algos, load_algos):
            if s != v and ('magnitude' in s and 'const' in v or 'const' in s and 'magnitude' in v):
                resume_ok = True

        # Priority mechanism ensures that algo permutations are irrelevant
        if set(save_algos) == set(load_algos):
            resume_ok = True
        else:
            saved_sparsity = filter(lambda x: x != QUANTIZATION, save_algos)
            loaded_sparsity = filter(lambda x: x != QUANTIZATION, load_algos)

            for s, v in zip(saved_sparsity, loaded_sparsity):
                # resume works fine for magnitude <-> const combo, because they have similar parameters
                if s != v and ('magnitude' in s and 'const' in v or 'const' in s and 'magnitude' in v):
                    resume_ok = True

    return {
        'save_algos': save_algos,
        'load_algos': load_algos,
        'is_resume_ok': resume_ok
    }


MODEL_WRAPPER = ["CPU", "GPU"]
WRAPPERS = list(itertools.product(MODEL_WRAPPER, MODEL_WRAPPER))


@pytest.fixture(scope='function', params=WRAPPERS,
                ids=['_'.join(['from:' + w[0], 'to:' + w[1]]) for w in WRAPPERS])
def _model_wrapper(request):
    modes = request.param

    def wrap_model(mode, model):
        if mode == "GPU":
            model = DataParallel(model, [0])
        return model

    return {
        'save_model': partial(wrap_model, modes[0]),
        'resume_model': partial(wrap_model, modes[1]),
    }


@pytest.mark.parametrize('is_resume', (True, False), ids=['resume', 'load_weights'])
def test_load_state_interoperability(_algos, _model_wrapper, is_resume):
    config_save = get_empty_config()
    config_save['compression'] = [{'algorithm': algo} for algo in _algos['save_algos']]
    register_bn_adaptation_init_args(config_save)
    compressed_model_save, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config_save)
    model_save = _model_wrapper['save_model'](compressed_model_save)
    saved_model_state = model_save.state_dict()
    ref_num_loaded = len(saved_model_state)

    config_resume = get_empty_config()
    config_resume['compression'] = [{'algorithm': algo} for algo in _algos['load_algos']]
    register_bn_adaptation_init_args(config_resume)
    compressed_model_resume, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(),
                                                                           config_resume)
    model_resume = _model_wrapper['resume_model'](compressed_model_resume)

    if not is_resume or (is_resume and _algos['is_resume_ok']):
        act_num_loaded = load_state(model_resume, saved_model_state, is_resume)

        if ('magnitude_sparsity' in _algos['load_algos'] or 'const_sparsity' in _algos['load_algos']) \
            and 'rb_sparsity' in _algos['save_algos']:
            # no need to load _mask and _uniform
            ref_num_loaded -= 2
        assert act_num_loaded == ref_num_loaded
    else:
        with pytest.raises(RuntimeError):
            load_state(model_resume, saved_model_state, is_resume)


LIST_ALGOS = [None, QUANTIZATION]
LIST_ALGOS += SPARSITY_ALGOS  # 3S


@pytest.mark.parametrize('is_resume', (True, False), ids=['resume', 'load_weights'])
@pytest.mark.parametrize('algo', tuple(LIST_ALGOS))
def test_ordinary_load(algo, _model_wrapper, is_resume):
    config = get_empty_config()
    if algo:
        config['compression'] = {'algorithm': algo}
    register_bn_adaptation_init_args(config)

    compressed_model_save, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    model_save = _model_wrapper['save_model'](compressed_model_save)

    compressed_model_resume, _ = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    model_resume = _model_wrapper['resume_model'](compressed_model_resume)

    num_loaded = load_state(model_resume, model_save.state_dict(), is_resume)

    assert num_loaded == len(model_save.state_dict())


def test_can_export_compressed_model_with_input_output_names(tmp_path):
    test_path = str(tmp_path.joinpath('test.onnx'))
    target_input_names = ['input1', 'input2']
    target_output_names = ['output1', 'output2']

    model = BasicTestModelWithTwoInputOutput()
    config = get_basic_asym_quantization_config()

    config["input_info"] = [{'sample_size': [1, 1, 4, 4]}, {'sample_size': [1, 1, 4, 4]}]
    register_bn_adaptation_init_args(config)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    compression_ctrl.export_model(test_path, input_names=target_input_names,
                                  output_names=target_output_names)

    assert os.path.exists(test_path)

    onnx_model = onnx.load(test_path)
    # pylint: disable=no-member
    curr_input_names = [node.name for node in onnx_model.graph.input]
    curr_output_names = [node.name for node in onnx_model.graph.output]

    assert curr_input_names == target_input_names
    assert curr_output_names == target_output_names


def test_can_export_compressed_model_with_specified_domain_for_custom_ops(tmp_path):
    test_path = str(tmp_path.joinpath('test.onnx'))

    model = BasicTestModelWithTwoInputOutput()
    config = get_basic_asym_quantization_config()

    config["input_info"] = [{'sample_size': [1, 1, 4, 4]}, {'sample_size': [1, 1, 4, 4]}]
    register_bn_adaptation_init_args(config)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    compression_ctrl.export_model(test_path)

    assert os.path.exists(test_path)

    onnx_model = onnx.load(test_path)

    count_custom_ops = 0
    # pylint: disable=no-member
    for op_node in onnx_model.graph.node:
        if op_node.op_type == "FakeQuantize":
            assert op_node.domain == DOMAIN_CUSTOM_OPS_NAME
            count_custom_ops += 1

    assert count_custom_ops == 4


def change_compression_algorithms_order(config):
    # changes order of compression algorithms in config
    def shift_list(list_for_shift):
        shifted_list = [list_for_shift.pop()] + list_for_shift
        return shifted_list

    config_compression = list(config.get('compression', {}))
    shifted_config_compression = shift_list(config_compression)
    config.update({'compression': shifted_config_compression})
    return config


def get_basic_rb_sparsity_int8_config():
    config = get_basic_sparsity_config()
    config.update({
        "compression": [
            {
                "algorithm": "rb_sparsity",
                "sparsity_init": 0.02,
                "params":
                    {
                        "schedule": "polynomial",
                        "sparsity_target": 0.5,
                        "sparsity_target_epoch": 2,
                        "sparsity_freeze_epoch": 3
                    },
            },
            {
                "algorithm": "quantization"
            }
        ]
    }
    )
    return config


comp_loss_configs = [
    get_basic_rb_sparsity_int8_config(),
    change_compression_algorithms_order(get_basic_rb_sparsity_int8_config())
]


@pytest.mark.parametrize("config", comp_loss_configs,
                         ids=[reduce(lambda x, y: x + "_" + y.get("algorithm", ""), config.get('compression', []),
                                     'compression')
                              for config in comp_loss_configs])
@pytest.mark.skipif(not cuda.is_available(), reason="Since its GPU test, no need to run this without GPUs available")
def test_compression_loss_gpu_device_compatibility(config):
    model = BasicConvTestModel()
    model.to(cuda.current_device())
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_ctrl.loss()


@pytest.mark.parametrize('algo_name, target_device',
                         list(itertools.product(
                             list(COMPRESSION_ALGORITHMS.registry_dict.keys()),
                             list(x.value for x in HWConfigType))))
def test_target_device_is_propagated_to_algos(mocker, algo_name, target_device):
    if algo_name == NoCompressionAlgorithmBuilder.__name__:
        pytest.skip()
    model = BasicConvTestModel()
    config = NNCFConfig.from_dict({
        "input_info":
        {
            "sample_size": [1, 1, 32, 32],
        },
        "compression": {
            "algorithm": algo_name
        },
        "target_device": target_device
    })
    register_bn_adaptation_init_args(config)

    import nncf
    compression_builder_init_spy = mocker.spy(nncf.api.compression.CompressionAlgorithmBuilder, '__init__')
    create_compressed_model_and_algo_for_test(model, config)
    assert compression_builder_init_spy.call_args[0][1]["hw_config_type"] == HWConfigType.from_str(target_device)
