import copy
import pytest
import numpy as np

from abc import ABC
from abc import abstractmethod
from typing import Callable, Dict, Generator, List, Optional, Tuple, TypeVar

from nncf import NNCFConfig
from tests.common.helpers import BaseTensorListComparator
from tests.common.helpers import TensorType

ParameterType = TypeVar('ParameterType')
GradientType = TypeVar('GradientType')
ModelType = TypeVar('ModelType')
DatasetType = TypeVar('DatasetType')


def get_config_algos(config: NNCFConfig) -> List[Dict]:
    if isinstance(config['compression'], list):
        algos = config['compression']
    else:
        algos = [config['compression']]
    return algos


def add_multiplier_to_config(config: NNCFConfig,
                             local_multiplier: Optional[float] = None,
                             global_multiplier: Optional[float] = None) -> NNCFConfig:
    config = copy.deepcopy(config)

    if local_multiplier is not None:
        algos = get_config_algos(config)
        for algo in algos:
            algo.update({
                'compression_lr_multiplier': local_multiplier
            })

    if global_multiplier is not None:
        config['compression_lr_multiplier'] = global_multiplier

    return config


def get_multipliers_from_config(config: NNCFConfig) -> Dict[str, float]:
    algo_to_multipliers = {}

    algos = get_config_algos(config)
    global_multiplier = config.get('compression_lr_multiplier', 1)
    for algo in algos:
        algo_name = algo['algorithm']
        algo_to_multipliers[algo_name] = algo.get('compression_lr_multiplier', global_multiplier)

    return algo_to_multipliers


def merge_configs(configs: List[NNCFConfig], use_algo_list: bool = True) -> NNCFConfig:
    res_config = NNCFConfig({})
    algos = []

    for source_config in configs:
        source_config = copy.deepcopy(source_config)
        algos.extend(get_config_algos(source_config))
        del source_config['compression']
        res_config.update(source_config)

    if not use_algo_list:
        if len(algos) > 1:
            raise Exception('If there is more than one algorithm '
                            'you could use only use_algo_list=True')
        res_config['compression'] = algos[0]
    else:
        res_config['compression'] = algos

    res_config['model'] = 'merged_model'
    return res_config


def get_quantization_config(sample_size: List[int]) -> NNCFConfig:
    config = NNCFConfig()
    config.update({
        'model': 'basic_quantization_config',

        'input_info': {
            'sample_size': sample_size,
        },

        'compression': {
            'algorithm': 'quantization',
            'initializer': {
                'range': {
                    'num_init_samples': 10,
                },
                'batchnorm_adaptation': {
                    'num_bn_adaptation_samples': 0,
                },
            },
        },
    })
    return config


def get_rb_sparsity_config(sample_size: List[int]) -> NNCFConfig:
    config = NNCFConfig()
    config.update({
        'model': 'basic_rb_sparsity_config',

        'input_info': {
            'sample_size': sample_size,
        },

        'compression': {
            'algorithm': 'rb_sparsity',
            'sparsity_init': 0.1,
            'params': {
                'schedule': 'polynomial',
                'sparsity_target': 0.5,
                'sparsity_target_epoch': 1,
                'sparsity_freeze_epoch': 1,
            },
        },
    })
    return config


def get_binarization_config(sample_size: List[int]) -> NNCFConfig:
    config = NNCFConfig()
    config.update({
        'model': 'basic_binarization_config',

        'input_info': {
            'sample_size': sample_size,
        },

        'compression': {
            'algorithm': 'binarization',
            'mode': 'xnor',
            'params': {
                'activations_quant_start_epoch': 0,
                'weights_quant_start_epoch': 0,
            },
        },
    })
    return config


def get_configs_building_params(get_orig_config_fns) -> List[Dict]:
    res = []
    num_orig_configs = len(get_orig_config_fns)

    for global_multiplier in [0, 1, 10]:
        res.append({
                'get_orig_config_fns': get_orig_config_fns,
                'multipliers': [None] * num_orig_configs,
                'global_multiplier': global_multiplier,
                'use_algo_list': True
        })

    global_multiplier = 10
    multipliers = [global_multiplier * (1.1 ** i) for i in range(num_orig_configs)]

    res.append({
        'get_orig_config_fns': get_orig_config_fns,
        'multipliers': multipliers,
        'global_multiplier': global_multiplier,
        'use_algo_list': True
    })

    for i in range(num_orig_configs):
        cur_multipliers = copy.deepcopy(multipliers)
        cur_multipliers[i] = None
        res.append({
            'get_orig_config_fns': get_orig_config_fns,
            'multipliers': cur_multipliers,
            'global_multiplier': None,
            'use_algo_list': True
        })

    for get_orig_config_fn in get_orig_config_fns:
        for use_algo_list in [False, True]:
            for global_multiplier, multiplier in [(11, 10), (11, None), (None, 10)]:
                res.append({
                    'get_orig_config_fns': [get_orig_config_fn],
                    'multipliers': [multiplier],
                    'global_multiplier': global_multiplier,
                    'use_algo_list': use_algo_list
                })

    return res


class BaseCompressionLRMultiplierTester(ABC):
    ALGO_NAME_TO_PATH_MAP = {}
    TensorListComparator: BaseTensorListComparator

    @pytest.fixture(name='ref_configs')
    def ref_configs_(self, configs_building_params: Dict, sample_size: Tuple[int, ...]) -> List[NNCFConfig]:
        return [get_ref_config_fn(sample_size) for get_ref_config_fn in configs_building_params['get_orig_config_fns']]

    @pytest.fixture(name='ref_config')
    def ref_config_(self, ref_configs, configs_building_params) -> NNCFConfig:
        return merge_configs(ref_configs, configs_building_params['use_algo_list'])

    @pytest.fixture(name='target_configs')
    def target_configs_(self, ref_configs: List[NNCFConfig], configs_building_params: Dict) -> List[NNCFConfig]:
        return [add_multiplier_to_config(config, local_multiplier=multiplier)
                for config, multiplier in zip(ref_configs, configs_building_params['multipliers'])]

    @pytest.fixture(name='target_config')
    def target_config_(self, target_configs: List[NNCFConfig], configs_building_params: Dict) -> NNCFConfig:
        target_config = merge_configs(target_configs, configs_building_params['use_algo_list'])
        return add_multiplier_to_config(target_config, global_multiplier=configs_building_params['global_multiplier'])

    @classmethod
    @abstractmethod
    def _get_layer_cls_and_params(cls, model: ModelType) -> Generator[Tuple[type, List[ParameterType]], None, None]:
        pass

    @classmethod
    def _get_params_grouped_by_algos(cls, model: ModelType) -> Dict[str, List[ParameterType]]:
        algo_name_to_params = {}
        for layer_cls, layer_params in cls._get_layer_cls_and_params(model):
            if len(layer_params) == 0:
                continue
            cls_name = '.'.join([layer_cls.__module__, layer_cls.__name__])

            for cur_algo_name, cur_algo_path in cls.ALGO_NAME_TO_PATH_MAP.items():
                if cur_algo_path in cls_name:
                    algo_name = cur_algo_name
                    break
            else:
                algo_name = 'regular'

            if algo_name not in algo_name_to_params:
                algo_name_to_params[algo_name] = []
            algo_name_to_params[algo_name].extend(layer_params)

        return algo_name_to_params

    @classmethod
    @abstractmethod
    def _get_params_and_grads_after_training_steps(cls, model: ModelType, dataset: DatasetType,
                                                   num_steps: int = 1) -> Tuple[Dict[str, ParameterType],
                                                                                Dict[str, GradientType]]:
        pass

    def test_algos_in_config_add_params(
            self,
            get_ref_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            target_config: NNCFConfig
    ):
        algo_to_params, _algo_to_grads = \
            self._get_params_and_grads_after_training_steps(*get_ref_model_and_dataset(), num_steps=0)
        algo_names = get_multipliers_from_config(target_config).keys()

        assert sorted(algo_to_params.keys()) == sorted(list(algo_names) + ['regular'])

    @classmethod
    def _check_multipliers_in_config_multiplies_grads(
            cls,
            get_ref_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            get_target_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            multipliers: Dict[str, float]
    ):
        _ref_params, ref_grads = \
            cls._get_params_and_grads_after_training_steps(*get_ref_model_and_dataset())
        _target_params, target_grads = \
            cls._get_params_and_grads_after_training_steps(*get_target_model_and_dataset())

        for algo in ref_grads:
            cls.TensorListComparator.check_equal([multipliers[algo] * grad for grad in ref_grads[algo]],
                                                 target_grads[algo])

    def test_multipliers_in_config_multiplies_grads(
            self,
            get_ref_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            get_target_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            target_config: NNCFConfig
    ):
        multipliers = get_multipliers_from_config(target_config)
        multipliers['regular'] = 1

        self._check_multipliers_in_config_multiplies_grads(get_ref_model_and_dataset,
                                                           get_target_model_and_dataset,
                                                           multipliers)

    @classmethod
    def _check_zero_multiplier_freezes_training(cls, orig_params: List[ParameterType], params: List[ParameterType],
                                                multiplier: float):
        if multiplier == 0:
            cls.TensorListComparator.check_equal(orig_params, params)
        else:
            cls.TensorListComparator.check_not_equal(orig_params, params)

    @classmethod
    def _get_params_diff(cls, orig_params: List[ParameterType], params: List[ParameterType]) -> List[TensorType]:
        param_diffs = []
        for param, orig_param in zip(params, orig_params):
            param_diffs.append(np.abs(param - orig_param))
        return param_diffs

    @classmethod
    def _check_multiplier_affects_training_speed(cls, orig_params: List[ParameterType],
                                                 ref_params: List[ParameterType], target_params: List[ParameterType],
                                                 multiplier: float):
        assert len(ref_params) == len(orig_params)
        assert len(target_params) == len(orig_params)

        ref_diff = cls._get_params_diff(ref_params, orig_params)
        target_diff = cls._get_params_diff(target_params, orig_params)

        if pytest.approx(multiplier) == 1:
            cls.TensorListComparator.check_equal(target_diff, ref_diff)
        elif multiplier < 1:
            cls.TensorListComparator.check_less(target_diff, ref_diff)
        else:
            cls.TensorListComparator.check_greater(target_diff, ref_diff)

    def _check_multipliers_in_config_affect_training_speed(
            self,
            get_ref_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            get_target_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            multipliers
    ):
        orig_params, _orig_grads = \
            self._get_params_and_grads_after_training_steps(*get_ref_model_and_dataset(), num_steps=0)
        ref_params, _ref_grads = \
            self._get_params_and_grads_after_training_steps(*get_ref_model_and_dataset(), num_steps=1)
        target_params, _target_grads = \
            self._get_params_and_grads_after_training_steps(*get_target_model_and_dataset(), num_steps=1)

        for algo in multipliers:
            self._check_zero_multiplier_freezes_training(orig_params[algo], target_params[algo], multipliers[algo])
            self._check_multiplier_affects_training_speed(
                orig_params[algo], ref_params[algo], target_params[algo], multipliers[algo]
            )

    def test_multipliers_in_config_affect_training_speed(
            self,
            get_ref_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            get_target_model_and_dataset: Callable[[], Tuple[ModelType, DatasetType]],
            target_config: NNCFConfig,
    ):
        multipliers = get_multipliers_from_config(target_config)
        multipliers['regular'] = 1

        self._check_multipliers_in_config_affect_training_speed(get_ref_model_and_dataset,
                                                                get_target_model_and_dataset,
                                                                multipliers)
