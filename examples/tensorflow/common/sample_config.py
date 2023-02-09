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

from addict import Dict

import argparse
import os

from examples.common.sample_config import SampleConfig
from nncf import NNCFConfig
from nncf.config.utils import is_experimental_quantization

try:
    import jstyleson as json
except ImportError:
    import json


def create_sample_config(args, parser) -> SampleConfig:
    sample_config = SampleConfig.from_json(args.config)
    sample_config.update_from_args(args, parser)

    nncf_config = NNCFConfig.from_json(args.config)

    if args.disable_compression and 'compression' in nncf_config:
        del nncf_config['compression']

    if 'compression' in nncf_config and is_experimental_quantization(nncf_config):
        from nncf.experimental.tensorflow.patch_tf import patch_tf_operations
        patch_tf_operations()

    sample_config.nncf_config = nncf_config

    return sample_config
