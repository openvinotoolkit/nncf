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

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import openvino.runtime as ov
from openvino.tools import pot

from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.logger import logger as nncf_logger
from nncf.data import Dataset
from nncf.openvino.engine import OVEngine
from nncf.openvino.utils import POTDataLoader
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice


def _convert_openvino_model_to_compressed_model(model: ov.Model,
                                                target_device: str) -> pot.graph.nx_model.CompressedModel:
    """
    Serializes the provided OpenVINO model and loads the model in the POT representation.

    :param model: The OpenVINO model.
    :param target_device: The target device.
    :return: The POT representation of the provided model.
    """
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp_dir:
        xml_path = str(Path(tmp_dir) / 'model.xml')
        bin_path = str(Path(tmp_dir) / 'model.bin')
        ov.serialize(model, xml_path, bin_path)
        model_config = {
            'model_name': 'model',
            'model': xml_path,
            'weights': bin_path,
        }
        pot_model = pot.load_model(model_config, target_device)

    return pot_model


def _convert_compressed_model_to_openvino_model(model: pot.graph.nx_model.CompressedModel) -> ov.Model:
    """
    Saves the provided POT compressed model and loads it as `openvino.runtime.Model` object.

    :param model: The POT compressed model.
    :return: The `openvino.runtime.Model`  object which represents the provided model.
    """
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp_dir:
        paths = pot.save_model(model, save_path=tmp_dir, model_name='model')
        xml_path = paths[0]['model']
        bin_path = paths[0]['weights']
        ie = ov.Core()
        ov_model = ie.read_model(xml_path, bin_path)
    return ov_model


def _create_ignored_scope_config(ignored_scope: Optional[IgnoredScope]) -> Dict:
    """
    Maps the content of `IgnoredScope` class to the `ignored` section of POT config.

    :param ignored_scope: The ignored scope
    :return: A POT ignored scope configuration as dict
    """
    if ignored_scope is None:
        return {}

    ignored = {}
    if ignored_scope.names is not None:
        ignored['scope'] = ignored_scope.names
    if ignored_scope.patterns is not None:
        raise RuntimeError('Quantization algorithm from the OpenVINO backend '
                           'does not support regular expressions in the ignored '
                           'scopes yet')
    if ignored_scope.types is not None:
        ignored['operations'] = [{'type': type} for type in ignored_scope.types]
    return ignored


def quantize_impl(model: ov.Model,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[ModelType] = None,
                  ignored_scope: Optional[IgnoredScope] = None) -> ov.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend.
    """
    pot.utils.logger.init_logger(
        level=logging.getLevelName(nncf_logger.getEffectiveLevel())
    )
    pot_model = _convert_openvino_model_to_compressed_model(model, target_device)

    engine_config = {
        'device': 'CPU',
        'stat_requests_number': 2,
        'eval_requests_number': 2,
    }

    algorithms = [
        {
            'name': 'DefaultQuantization',
            'params': {
                'target_device': target_device.value,
                'preset': preset.value,
                'stat_subset_size': subset_size,
                'use_fast_bias': fast_bias_correction,
                'model_type': None if model_type is None else model_type.value,
                'ignored': _create_ignored_scope_config(ignored_scope)
            }
        }
    ]

    pot_dataloader = POTDataLoader(calibration_dataset)
    engine = OVEngine(engine_config, pot_dataloader, pot_dataloader)
    pipeline = pot.create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(pot_model)
    pot.compress_model_weights(compressed_model)

    quantized_model = _convert_compressed_model_to_openvino_model(compressed_model)

    return quantized_model
