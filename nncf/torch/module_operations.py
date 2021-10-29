"""
 Copyright (c) 2019 Intel Corporation
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

from torch import nn

from nncf.torch.layers import NNCF_PADDING_VALUE_ATTR_NAME


class BaseOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    @property
    def operand(self):
        return self.op

    def forward(self, *inputs, **kwargs):
        return self.op(*inputs, **kwargs)


class UpdateInputs(BaseOp):
    def __call__(self, _, inputs):
        return super().__call__(*inputs)


class UpdateParameter(BaseOp):
    def __init__(self, param_name, op):
        super().__init__(op)
        self._param_name = param_name

    def __call__(self, module, _):
        if not hasattr(module, self._param_name):
            raise TypeError('{} should have {} attribute'.format(type(module), self._param_name))
        value = getattr(module, self._param_name)
        result = super().__call__(value)
        setattr(module, self._param_name, result)


class UpdateWeight(UpdateParameter):
    def __init__(self, op):
        super().__init__("weight", op)


class UpdateParameterList(BaseOp):
    """
    A module which updates attributes of a module fed to
    forward method call by operand call.
    """

    def __init__(self, param_names, op):
        super().__init__(op)
        self._param_names = param_names

    def __call__(self, module, _):
        param_values = []
        for param_name in self._param_names:
            if not hasattr(module, param_name):
                raise TypeError('{} should have {} attribute'.format(type(module), param_name))
            param_values.append(getattr(module, param_name))
        updated_kwargs = dict(zip(self._param_names, param_values))
        updated_values = super().__call__(**updated_kwargs)

        for param_name, updated_value in zip(self._param_names, updated_values):
            setattr(module, param_name, updated_value)


class UpdateWeightAndBias(UpdateParameterList):
    """
    A module which updates `weight` and `bias` attributes of a module
    fed to forward method call by operand call.
    """

    def __init__(self, op):
        super().__init__(["weight", "bias"], op)


class UpdatePaddingValue(UpdateParameter):
    def __init__(self, op):
        super().__init__(NNCF_PADDING_VALUE_ATTR_NAME, op)
