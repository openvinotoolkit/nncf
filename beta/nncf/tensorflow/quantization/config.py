"""
 Copyright (c) 2020 Intel Corporation
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


class QuantizationMode:
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantizerConfig:
    def __init__(self, mode=QuantizationMode.SYMMETRIC,
                 num_bits=8,
                 signed=None,
                 per_channel=False,
                 narrow_range=False):
        self.num_bits = num_bits
        self.mode = mode
        self.signed = signed
        self.per_channel = per_channel
        self.narrow_range = narrow_range

    def __str__(self):
        return "B:{num_bits} M:{mode} SGN:{signed} NR:{narrow_range} PC:{per_channel}".format(
            num_bits=self.num_bits,
            mode='S' if self.mode == QuantizationMode.SYMMETRIC else 'A',
            signed='ANY' if self.signed is None else ('S' if self.signed else 'U'),
            narrow_range='Y' if self.narrow_range else 'N',
            per_channel='Y' if self.per_channel else 'N')

    def __hash__(self):
        return hash(str(self))


class QuantizationConstraints:
    def __init__(self, mode=None,
                 num_bits=None,
                 signed=None,
                 per_channel=None,
                 narrow_range=None):
        self.constraints = {}
        self._initialize_constrains(mode=mode, num_bits=num_bits, signed=signed,
                                    per_channel=per_channel, narrow_range=narrow_range)

    def apply_constraints_to(self, qconfig: QuantizerConfig) -> QuantizerConfig:
        for attr_name, constraint in self.constraints.items():
            setattr(qconfig, attr_name, constraint)
        return qconfig

    def is_config_compatible(self, qconfig: QuantizerConfig) -> bool:
        is_compatible = True
        for attr_name, constraint in self.constraints.items():
            qconf_attr_value = getattr(qconfig, attr_name)
            if qconf_attr_value != constraint:
                is_compatible = False
        return is_compatible

    def _initialize_constrains(self, **kwargs):
        for name, value in kwargs.items():
            if value is not None:
                self.constraints[name] = value
