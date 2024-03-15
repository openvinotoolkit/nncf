# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from copy import deepcopy
from pathlib import Path

import yaml
from addict import Dict

try:
    from accuracy_checker.config import ConfigReader
except ImportError:
    from openvino.tools.accuracy_checker.config import ConfigReader

import nncf
from nncf.common.logging import nncf_logger

DEFAULT_PATH = "PATH"
DEFAULT_TARGET_DEVICE = "ANY"


class Config(Dict):
    """
    Tool configuration containing model, engine, and algorithms' parameters.
    Adapted from openvino/tools/pot/configs/config.py.
    """

    @staticmethod
    def _read_config_from_file(path):
        return read_config_from_file(path)

    @classmethod
    def read_config(cls, path):
        data = cls._read_config_from_file(path)
        config = cls(data)
        return config

    def configure_params(self, ac_config=None):
        if ac_config is not None:
            self.engine["config"] = ac_config

        self._configure_engine_params()
        self._convert_mo_args()
        self._configure_algo_params()
        self._configure_logger_params()

    def add_log_dir(self, model_log_dir, exec_log_dir):
        self.model["model_log_dir"] = model_log_dir
        self.model["exec_log_dir"] = exec_log_dir
        if "optimizer" in self:
            self.optimizer.params["model_log_dir"] = model_log_dir
            self.optimizer.params["exec_log_dir"] = exec_log_dir
        if "compression" in self:
            for algo_config in self.compression.algorithms:
                algo_config.params["exec_log_dir"] = exec_log_dir

    def _convert_mo_args(self):
        """Converts model args into model optimizer parameters"""
        config = self.model
        self._validate_model_conf()
        if not self.model.model_name:
            self.model.model_name = Path(self.model.model).stem
        self["model"] = Config(config)

    def _validate_model_conf(self):
        """Validates the correctness of model config"""
        models = self.model.cascade if self.model.cascade else [self.model]

        if self.model.cascade and len(models) == 1:
            nncf_logger.warning("Cascade is defined with single model")

        if not models:
            raise nncf.ValidationError("Path to input model xml and bin is required.")

        for model in models:
            if len(models) > 1 and not model.name:
                raise nncf.ValidationError("Name of input model is required.")
            if not model.model:
                raise nncf.ValidationError("Path to input model xml is required.")
            if not model.weights:
                raise nncf.ValidationError("Path to input model bin is required.")

    def validate_algo_config(self):
        """
        Validates the correctness of algorithm parameters in config
        """
        range_estimator_parameters = {
            "preset": None,
            "min": {"type": None, "outlier_prob": None, "granularity": None, "clipping_value": None},
            "max": {"type": None, "outlier_prob": None, "granularity": None, "clipping_value": None},
        }
        weights_params = {
            "bits": None,
            "mode": None,
            "level_low": None,
            "level_high": None,
            "granularity": None,
            "range_estimator": range_estimator_parameters,
        }
        activations_params = deepcopy(weights_params)
        activations_params["range_estimator"]["min"].update({"aggregator": None})
        activations_params["range_estimator"]["max"].update({"aggregator": None})

        ignored = {"ignored": {}}
        ignored_content = {"skip_model": None, "scope": None, "operations": None}
        if self.model.cascade:
            for model in self.model.cascade:
                ignored["ignored"].update({model.name: ignored_content})
        else:
            ignored["ignored"] = ignored_content

        bias_correction_params = {
            "stat_subset_size": None,
            "shuffle_data": None,
            "seed": None,
            "apply_for_all_nodes": None,
            "threshold": None,
        }

        layerwise_finetuning_params = {
            "num_samples_for_tuning": None,
            "batch_size": None,
            "optimizer": None,
            "loss": None,
            "tuning_iterations": None,
            "random_seed": None,
            "use_ranking_subset": None,
            "calibration_indices_pool": None,
            "calculate_grads_on_loss_increase_only": None,
            "weight_decay": None,
        }

        supported_params = {
            "ActivationChannelAlignment": {"stat_subset_size": None, "shuffle_data": None, "seed": None},
            "MinMaxQuantization": {
                "preset": None,
                "stat_subset_size": None,
                "shuffle_data": None,
                "seed": None,
                "range_estimator": range_estimator_parameters,
                "weights": weights_params,
                "activations": activations_params,
                "saturation_fix": None,
            },
            "FastBiasCorrection": bias_correction_params,
            "BiasCorrection": bias_correction_params,
            "DefaultQuantization": {"use_fast_bias": None, "use_layerwise_tuning": None},
            "ParamsTuningSearch": {},
            "DataFreeQuantization": {"preset": None, "weights": weights_params, "activations": activations_params},
            "AccuracyAwareQuantization": {
                "metric_subset_ratio": None,
                "ranking_subset_size": None,
                "max_iter_num": None,
                "maximal_drop": None,
                "drop_type": None,
                "use_prev_if_drop_increase": None,
                "metrics": None,
                "base_algorithm": "DefaultQuantization",
                "annotation_free": None,
                "tune_hyperparams": None,
                "annotation_conf_threshold": None,
                "convert_to_mixed_preset": None,
            },
            "RangeOptimization": {
                "stat_subset_size": None,
                "result_filename": None,
                "maxiter": None,
                "lower_boxsize": None,
                "upper_boxsize": None,
                "zero_boxsize": None,
                "optimization_scope": None,
                "activation_ranges_to_set": None,
                "metric_name": None,
                "optimizer_name": None,
                "stochastic": None,
                "dump_model_prefix": None,
                "error_function": None,
                "opt_backend": None,
            },
            "TunableQuantization": {"outlier_prob_choices": None},
            "MagnitudeSparsity": {
                "sparsity_level": None,
                "normed_threshold": None,
            },
            "BaseWeightSparsity": {
                "use_fast_bias": None,
            },
            "WeightSparsity": {
                "use_layerwise_tuning": None,
            },
            "OverflowCorrection": {
                "stat_subset_size": None,
                "shuffle_data": None,
                "seed": None,
            },
            "RangeSupervision": {
                "stat_subset_size": None,
                "shuffle_data": None,
                "seed": None,
            },
        }

        # completing supported parameters
        for algo_name in supported_params:
            supported_params[algo_name].update(ignored)

        for algo_name in ["DefaultQuantization", "WeightSparsity"]:
            supported_params[algo_name].update(layerwise_finetuning_params)

        for algo_name in ["ActivationChannelAlignment", "MinMaxQuantization", "FastBiasCorrection", "BiasCorrection"]:
            supported_params["DefaultQuantization"].update(supported_params[algo_name])
            supported_params["ParamsTuningSearch"].update(supported_params[algo_name])

        for algo_name in ["MagnitudeSparsity", "FastBiasCorrection"]:
            supported_params["BaseWeightSparsity"].update(supported_params[algo_name])

        for algo_name in ["BaseWeightSparsity"]:
            supported_params["WeightSparsity"].update(supported_params[algo_name])

        supported_params["TunableQuantization"].update(supported_params["MinMaxQuantization"])

        # check algorithm parameters
        for algo in self["compression"]["algorithms"]:
            algo_name = algo["name"]
            if algo_name in supported_params:
                if algo_name == "AccuracyAwareQuantization":
                    backup = deepcopy(supported_params["AccuracyAwareQuantization"])
                    base_algo = supported_params["AccuracyAwareQuantization"]["base_algorithm"]
                    if "base_algorithm" in algo["params"] and algo["params"]["base_algorithm"] in supported_params:
                        base_algo = algo["params"]["base_algorithm"]
                    supported_params["AccuracyAwareQuantization"].update(supported_params[base_algo])

                check_params(algo_name, algo["params"], supported_params[algo_name])

                if algo_name == "AccuracyAwareQuantization":
                    supported_params["AccuracyAwareQuantization"] = backup

    def _configure_engine_params(self):
        """Converts engine config section into engine params dict"""
        engine = self.engine
        if "type" not in engine or engine.type == "accuracy_checker":
            self._configure_ac_params()
            self.engine.type = "accuracy_checker"
        elif engine.type == "simplified":
            if engine.data_source is None:
                raise KeyError("Missed data dir for simplified engine")
            self.engine.device = engine.device if engine.device else "CPU"
            engine.data_source = Path(engine.data_source)
        else:
            raise KeyError("Unsupported engine type")

    def _configure_ac_params(self):
        """Converts engine config into accuracy checker config"""
        filtering_params = {"target_devices": ["CPU"], "target_framework": "openvino", "use_new_api": True}
        if "config" in self.engine:
            ac_conf, mode = ConfigReader.merge(Dict({**self.engine, **filtering_params}))
            ac_conf = Dict(ac_conf)
        else:
            mode = "evaluations" if self.engine.module else "models"
            ac_conf = Dict({mode: [self.engine]})
            ac_conf[mode][0].name = self.model.model_name
            datasets_config = ac_conf[mode][0].datasets if mode == "models" else ac_conf[mode][0].module_config.datasets
            if isinstance(datasets_config, dict) and "preprocessing" in datasets_config:
                nncf_logger.debug("Global dataset preprocessing configuration found")
                preprocessing_config = datasets_config.pop("preprocessing")
                for dataset_name, dataset in datasets_config.items():
                    if not dataset.preprocessing:
                        dataset["preprocessing"] = preprocessing_config
                    else:
                        nncf_logger.debug(
                            "Local preprocessing configuration is used for {} dataset".format(dataset_name)
                        )
            ConfigReader.check_local_config(ac_conf)
            ac_conf = ConfigReader.convert_paths(ac_conf)
            ConfigReader._filter_launchers(ac_conf, filtering_params, mode=mode)
        for req_num in ["stat_requests_number", "eval_requests_number"]:
            ac_conf[req_num] = self.engine[req_num] if req_num in self.engine else None

        self["engine"] = ac_conf

    def _configure_algo_params(self):
        """Converts algorithm params into valid configuration for algorithms"""
        self.validate_algo_config()

        for algo in self["compression"]["algorithms"]:
            preset = algo["params"].get("preset", "performance")
            aliases = {"symmetric": "performance", "asymmetric": "accuracy"}
            preset = aliases.get(preset, preset)
            presets_aliases_by_device = {
                "NPU": {"accuracy": "accuracy"},
                "GNA": {"accuracy": "accuracy", "mixed": "accuracy"},
                "GNA3": {"accuracy": "accuracy", "mixed": "accuracy"},
                "GNA3.5": {"accuracy": "accuracy", "mixed": "accuracy"},
                "CPU": {"accuracy": "mixed"},
                "ANY": {"accuracy": "mixed"},
                "GPU": {"accuracy": "mixed"},
            }
            algo["params"]["target_device"] = self["compression"].get("target_device", DEFAULT_TARGET_DEVICE)
            rename = presets_aliases_by_device.get(algo["params"]["target_device"], {"accuracy": "mixed"})
            algo["params"]["preset"] = rename.get(preset, preset)
            model_type = self["compression"].get("model_type", None)
            algo["params"]["model_type"] = None if model_type == "None" else model_type
            algo["params"]["dump_intermediate_model"] = self["compression"].get("dump_intermediate_model", False)
            algo["params"]["inplace_statistics"] = self["compression"].get("inplace_statistics", True)

    def _configure_logger_params(self):
        """Creates a log directory name based on model and algo configurations"""

        # init log folder name
        log_algo_name = self.model.model_name
        if "optimizer" in self:
            log_algo_name = "{}_{}".format(log_algo_name, self["optimizer"]["name"])
        for algo in self["compression"]["algorithms"]:
            log_algo_name = ("{}_{}".format(log_algo_name, algo.name)) if log_algo_name else algo.name
        self.model.log_algo_name = log_algo_name

    def get_model_paths(self):
        """Returns models paths
        :return: dictionary with model tokens as keys and
        dictionaries of paths to xml and bin as values
        """
        return (
            deepcopy(self.model.cascade)
            if self.model.cascade
            else [{"model": self.model.model, "weights": self.model.weights}]
        )


def read_config_from_file(path):
    path = Path(path)
    extension = path.suffix.lower()
    with path.open() as f:
        if extension in (".yaml", ".yml"):
            return yaml.load(f, Loader=yaml.SafeLoader)
        if extension in (".json",):
            return json.load(f)
        raise nncf.InternalError('Unknown file extension for the file "{}"'.format(path))


def check_params(algo_name, config, supported_params):
    """Check algorithm parameters in config
    :param algo_name: name of algorithm
    :param config: config with parameters to check
    :param supported_params: parameters supported by algorithm
    """
    for key, value in config.items():
        if key not in supported_params:
            raise nncf.InternalError("Algorithm {}. Unknown parameter: {}".format(algo_name, key))
        if isinstance(value, dict):
            if isinstance(supported_params[key], dict):
                check_params(algo_name, value, supported_params[key])
            else:
                raise nncf.InternalError("Algorithm {}. Wrong structure for parameter: {}".format(algo_name, key))
