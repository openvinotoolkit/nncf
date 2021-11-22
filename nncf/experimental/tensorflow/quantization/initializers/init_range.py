"""
 Copyright (c) 2021 Intel Corporation
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

from itertools import islice

from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.utils.progress_bar import ProgressBar
from nncf.tensorflow.quantization.initializers.init_range import TFRangeInitParams
from nncf.tensorflow.quantization.initializers.init_range import RangeInitializer

from nncf.experimental.tensorflow.quantization.quantizers import InputType
from nncf.experimental.tensorflow.quantization.quantizers import NNCF_QUANTIZATION_OPERATIONS
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork


class TFRangeInitParamsV2(TFRangeInitParams):
    def get_init_config_for_quantization_point(self,
                                               node_name: str,
                                               input_type: InputType) -> RangeInitConfig:
        group = QuantizerGroup.WEIGHTS if input_type == InputType.WEIGHTS else QuantizerGroup.ACTIVATIONS
        return self.get_init_config_for_scope_and_group(node_name, group)


class RangeInitializerV2(RangeInitializer):
    def __init__(self, range_init_params: TFRangeInitParamsV2):
        self.range_init_params = range_init_params
        self.dataset = range_init_params.init_range_data_loader
        self.num_steps = range_init_params.get_max_num_init_steps()

        self.nncf_quantization_operation_classes = NNCF_QUANTIZATION_OPERATIONS.registry_dict.values()

    def run(self, model: NNCFNetwork) -> None:
        handles = []
        collectors = []
        for op in model.nncf_operations:
            if op.__class__ not in self.nncf_quantization_operation_classes:
                continue

            # TODO(andrey-churkin): Use correct node name
            node_name = ''
            init_config = self.range_init_params.get_init_config_for_quantization_point(node_name, op.input_type)
            collector = RangeInitializerV2.generate_stat_collector(
                init_config,
                op.per_channel,
                op.channel_axes,
                op.input_type.value
            )
            handles.append(op.register_hook_pre_call(collector))
            op.enabled = False
            collectors.append((op, collector))

        for (x, _) in ProgressBar(
                islice(self.dataset, self.num_steps),
                total=self.num_steps,
                desc='Collecting tensor statistics/data'
        ):
            model(x, training=False)

        for op, collector in collectors:
            collector.prepare_statistics()
            op.apply_range_initialization(collector.min, collector.max)
            op.enabled = True

        for handle in handles:
            handle.remove()

        for x, _ in self.dataset:
            model(x, training=False)
            break
