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
import os
from typing import List

from nncf import __version__
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.utils.logger import logger as nncf_logger
from nncf.definitions import NNCF_CI_ENV_VAR_NAME
from nncf.definitions import NNCF_DEV_ENV_VAR_NAME

# MEASUREMENT_ID = 'G-4Z7Y9HHRMD'  # GA4
MEASUREMENT_ID = 'UA-242812675-1'

# For recommendations on proper usage of categories, actions, labels and values, see:
# https://support.google.com/analytics/answer/1033068
try:
    from openvino_telemetry import Telemetry
    NNCFTelemetry = Telemetry(app_name='nncf', app_version=__version__, tid=MEASUREMENT_ID)
except ImportError:
    nncf_logger.debug("openvino_telemetry package not found. No telemetry will be sent.")
    NNCFTelemetry = None

if os.getenv(NNCF_CI_ENV_VAR_NAME) or os.getenv(NNCF_DEV_ENV_VAR_NAME) or NNCFTelemetry is None:
    from unittest.mock import MagicMock
    NNCFTelemetry = MagicMock()

NNCF_TF_CATEGORY = 'nncf_tf'
NNCF_PT_CATEGORY = 'nncf_pt'
NNCF_ONNX_CATEGORY = 'nncf_onnx'
NNCF_OV_CATEGORY = 'nncf_ov'

def get_algo_names_from_builder(builder: CompressionAlgorithmBuilder) -> List[str]:
    if isinstance(builder, CompositeCompressionAlgorithmBuilder):
        algo_names = sorted([x.name for x in builder.child_builders])
    else:
        algo_names = [builder.name, ]
    return algo_names
