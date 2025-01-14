# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import numbers
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.weight_norm import WeightNorm

import nncf
from nncf import nncf_logger
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.utils.api_marker import api
from nncf.common.utils.registry import Registry
from nncf.torch.checkpoint_loading import OPTIONAL_PARAMETERS_REGISTRY
from nncf.torch.dynamic_graph.context import forward_nncf_trace
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.utils import no_jit_trace


def dict_update(src: Dict, dst: Dict, recursive: bool = True):
    for name, value in src.items():
        if recursive and name in dst and isinstance(value, dict):
            dict_update(value, dst[name], recursive)
        else:
            dst[name] = value


def maybe_reapply_weight_norm(src: torch.nn.Module, dst: torch.nn.Module) -> torch.nn.Module:
    for k, hook in src._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm):
            # The code below presumes that the `hook` object does not
            # contain internal references to the module it was set up on
            # (i.e. to the `src`) and takes the module to act on as a parameter.
            # This is the case for the `WeightNorm` hook.
            hook.remove(dst)
            del dst._forward_pre_hooks[k]
            name = hook.name
            dim = hook.dim
            WeightNorm.apply(dst, name=name, dim=dim)
    return dst


def align_module_internals(src: torch.nn.Module, dst: torch.nn.Module) -> torch.nn.Module:
    dict_update(src.__dict__, dst.__dict__)
    dst = maybe_reapply_weight_norm(src, dst)
    return dst


class NNCFConv1d(_NNCFModuleMixin, nn.Conv1d):
    op_func_name = "conv1d"

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv1d.__name__
        nncf_conv = NNCFConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            hasattr(module, "bias"),
        )
        nncf_conv = align_module_internals(module, nncf_conv)
        return nncf_conv


class NNCFConvTranspose1d(_NNCFModuleMixin, nn.ConvTranspose1d):
    op_func_name = "conv_transpose1d"
    target_weight_dim_for_compression = 1

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.ConvTranspose1d.__name__
        args = [
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.output_padding,
            module.groups,
            hasattr(module, "bias"),
            module.dilation,
        ]
        if hasattr(module, "padding_mode"):
            args.append(module.padding_mode)
        nncf_conv_transpose1d = NNCFConvTranspose1d(*args)
        nncf_conv_transpose1d = align_module_internals(module, nncf_conv_transpose1d)
        return nncf_conv_transpose1d


NNCF_PADDING_VALUE_ATTR_NAME = "nncf_padding_value"
OPTIONAL_PARAMETERS_REGISTRY.register(NNCF_PADDING_VALUE_ATTR_NAME)


class NNCFConv2d(_NNCFModuleMixin, nn.Conv2d):
    op_func_name = "conv2d"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.register_buffer(NNCF_PADDING_VALUE_ATTR_NAME, torch.zeros([1]))

    def get_padding_value_ref(self):
        return getattr(self, NNCF_PADDING_VALUE_ATTR_NAME)

    def _set_padding_value(self, value):
        setattr(self, NNCF_PADDING_VALUE_ATTR_NAME, value)

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv2d.__name__
        nncf_conv = NNCFConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            hasattr(module, "bias"),
        )
        nncf_conv = align_module_internals(module, nncf_conv)
        return nncf_conv

    # override class attribute of _NNCFModuleMixin
    def _custom_forward_fn(self, input_):
        proxy_padding_value = getattr(self, NNCF_PADDING_VALUE_ATTR_NAME)  # hack to get value from ProxyModule
        proxy_weight = self.weight
        proxy_bias = self.bias
        proxy_padding = self.padding
        proxy_num_groups = self.groups
        return self._conv_forward_proxy(
            input_, proxy_weight, proxy_bias, proxy_padding_value, proxy_padding, proxy_num_groups
        )

    def _conv_forward_proxy(self, input_, weight, bias, padding_value, padding, num_groups):
        with no_jit_trace():
            padding_val = padding_value.item()
        self.get_padding_value_ref().data.fill_(padding_val)
        self.groups = num_groups

        def _reverse_repeat_tuple(t, n):
            r"""Reverse the order of `t` and repeat each element for `n` times.

            This can be used to translate padding arg used by Conv and Pooling modules
            to the ones used by `F.pad`.
            """
            return tuple(x for x in reversed(t) for _ in range(n))

        reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input_, reversed_padding_repeated_twice, mode=self.padding_mode, value=padding_val),
                weight,
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        if padding_val == 0:
            return F.conv2d(input_, weight, bias, self.stride, padding, self.dilation, self.groups)
        return F.conv2d(
            F.pad(input_, reversed_padding_repeated_twice, value=padding_val),
            weight,
            bias,
            self.stride,
            (0, 0),
            self.dilation,
            self.groups,
        )


class NNCFLinear(_NNCFModuleMixin, nn.Linear):
    op_func_name = "linear"

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Linear.__name__

        nncf_linear = NNCFLinear(module.in_features, module.out_features, hasattr(module, "bias"))
        nncf_linear = align_module_internals(module, nncf_linear)
        return nncf_linear


class NNCFBatchNorm1d(_NNCFModuleMixin, nn.BatchNorm1d):
    op_func_name = "batch_norm"
    ignored_algorithms = ["magnitude_sparsity", "rb_sparsity", "const_sparsity", "quantization"]

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.BatchNorm1d.__name__

        nncf_bn = NNCFBatchNorm1d(module.num_features)
        nncf_bn = align_module_internals(module, nncf_bn)
        return nncf_bn


class NNCFBatchNorm2d(_NNCFModuleMixin, nn.BatchNorm2d):
    op_func_name = "batch_norm"
    ignored_algorithms = ["magnitude_sparsity", "rb_sparsity", "const_sparsity", "quantization"]

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.BatchNorm2d.__name__

        nncf_bn = NNCFBatchNorm2d(module.num_features)
        nncf_bn = align_module_internals(module, nncf_bn)
        return nncf_bn


class NNCFBatchNorm3d(_NNCFModuleMixin, nn.BatchNorm3d):
    op_func_name = "batch_norm"
    ignored_algorithms = ["magnitude_sparsity", "rb_sparsity", "const_sparsity", "quantization"]

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.BatchNorm3d.__name__

        nncf_bn = NNCFBatchNorm3d(module.num_features)
        nncf_bn = align_module_internals(module, nncf_bn)
        return nncf_bn


class NNCFGroupNorm(_NNCFModuleMixin, nn.GroupNorm):
    op_func_name = "group_norm"
    ignored_algorithms = ["magnitude_sparsity", "rb_sparsity", "const_sparsity", "quantization"]

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.GroupNorm.__name__

        nncf_bn = NNCFGroupNorm(
            num_groups=module.num_groups, num_channels=module.num_channels, eps=module.eps, affine=module.affine
        )
        nncf_bn = align_module_internals(module, nncf_bn)
        return nncf_bn


class NNCFLayerNorm(_NNCFModuleMixin, nn.LayerNorm):
    op_func_name = "layer_norm"
    ignored_algorithms = ["magnitude_sparsity", "rb_sparsity", "const_sparsity", "quantization"]

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.LayerNorm.__name__

        nncf_ln = NNCFLayerNorm(module.normalized_shape, hasattr(module, "eps"))
        nncf_ln = align_module_internals(module, nncf_ln)
        return nncf_ln


class NNCFConvTranspose2d(_NNCFModuleMixin, nn.ConvTranspose2d):
    op_func_name = "conv_transpose2d"
    target_weight_dim_for_compression = 1

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.ConvTranspose2d.__name__
        args = [
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.output_padding,
            module.groups,
            hasattr(module, "bias"),
            module.dilation,
        ]
        if hasattr(module, "padding_mode"):
            args.append(module.padding_mode)
        nncf_conv_transpose2d = NNCFConvTranspose2d(*args)
        nncf_conv_transpose2d = align_module_internals(module, nncf_conv_transpose2d)
        return nncf_conv_transpose2d


class NNCFConv3d(_NNCFModuleMixin, nn.Conv3d):
    op_func_name = "conv3d"

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv3d.__name__

        nncf_conv3d = NNCFConv3d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            hasattr(module, "bias"),
        )
        nncf_conv3d = align_module_internals(module, nncf_conv3d)
        return nncf_conv3d


class NNCFConvTranspose3d(_NNCFModuleMixin, nn.ConvTranspose3d):
    op_func_name = "conv_transpose3d"
    target_weight_dim_for_compression = 1

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.ConvTranspose3d.__name__
        args = [
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.output_padding,
            module.groups,
            hasattr(module, "bias"),
            module.dilation,
        ]
        if hasattr(module, "padding_mode"):
            args.append(module.padding_mode)
        nncf_conv_transpose3d = NNCFConvTranspose3d(*args)
        nncf_conv_transpose3d = align_module_internals(module, nncf_conv_transpose3d)
        return nncf_conv_transpose3d


class NNCFEmbedding(_NNCFModuleMixin, nn.Embedding):
    op_func_name = "embedding"
    target_weight_dim_for_compression = 0

    # Note that this does not require activation quantization because it's basically a lookup.
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Embedding.__name__

        args = [
            module.num_embeddings,
            module.embedding_dim,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.sparse,
            module.weight,
        ]
        nncf_embedding = NNCFEmbedding(*args)
        nncf_embedding = align_module_internals(module, nncf_embedding)
        return nncf_embedding


class NNCFEmbeddingBag(_NNCFModuleMixin, nn.EmbeddingBag):
    op_func_name = "embedding_bag"

    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.EmbeddingBag.__name__

        args = [
            module.num_embeddings,
            module.embedding_dim,
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.mode,
            module.sparse,
            module.weight,
            module.include_last_offset,
        ]
        nncf_embedding_bag = NNCFEmbeddingBag(*args)
        nncf_embedding_bag = align_module_internals(module, nncf_embedding_bag)
        return nncf_embedding_bag


NNCF_MODULES_DICT = {
    NNCFConv1d: nn.Conv1d,
    NNCFConv2d: nn.Conv2d,
    NNCFConv3d: nn.Conv3d,
    NNCFLinear: nn.Linear,
    NNCFBatchNorm1d: nn.BatchNorm1d,
    NNCFBatchNorm2d: nn.BatchNorm2d,
    NNCFBatchNorm3d: nn.BatchNorm3d,
    NNCFGroupNorm: nn.GroupNorm,
    NNCFLayerNorm: nn.LayerNorm,
    NNCFConvTranspose1d: nn.ConvTranspose1d,
    NNCFConvTranspose2d: nn.ConvTranspose2d,
    NNCFConvTranspose3d: nn.ConvTranspose3d,
    NNCFEmbedding: nn.Embedding,
    NNCFEmbeddingBag: nn.EmbeddingBag,
}

NNCF_MODULES_MAP = {k.__name__: v.__name__ for k, v in NNCF_MODULES_DICT.items()}
NNCF_MODULES = list(NNCF_MODULES_MAP.keys())
NNCF_MODULES_OP_NAMES = [k.op_func_name for k, _ in NNCF_MODULES_DICT.items()]

NNCF_CONV_MODULES_DICT = {
    NNCFConv1d: nn.Conv1d,
    NNCFConv2d: nn.Conv2d,
    NNCFConv3d: nn.Conv3d,
}
NNCF_DECONV_MODULES_DICT = {
    NNCFConvTranspose1d: nn.ConvTranspose1d,
    NNCFConvTranspose2d: nn.ConvTranspose2d,
    NNCFConvTranspose3d: nn.ConvTranspose3d,
}
NNCF_CONV_MODULES_MAP = {k.__name__: v.__name__ for k, v in NNCF_CONV_MODULES_DICT.items()}
NNCF_CONV_MODULES = list(NNCF_CONV_MODULES_MAP.keys())
NNCF_GENERAL_CONV_MODULES_DICT = dict(NNCF_CONV_MODULES_DICT)
NNCF_GENERAL_CONV_MODULES_DICT.update(NNCF_DECONV_MODULES_DICT)

NNCF_LINEAR_MODULES_DICT = {NNCFLinear: nn.Linear}
NNCF_MODULES_OP_NAMES = [k.op_func_name for k, _ in NNCF_MODULES_DICT.items()]

NNCF_PRUNING_MODULES_DICT = {
    NNCFLinear: nn.Linear,
    NNCFConv1d: nn.Conv1d,
    NNCFConv2d: nn.Conv2d,
    NNCFConv3d: nn.Conv3d,
    NNCFConvTranspose1d: nn.ConvTranspose1d,
    NNCFConvTranspose2d: nn.ConvTranspose2d,
    NNCFConvTranspose3d: nn.ConvTranspose3d,
}
NNCF_PRUNING_MODULES_MAP = {k.__name__: v.__name__ for k, v in NNCF_CONV_MODULES_DICT.items()}
NNCF_PRUNING_MODULES = list(NNCF_CONV_MODULES_MAP.keys())

UNWRAPPED_USER_MODULES = Registry("user_modules")
NNCF_WRAPPED_USER_MODULES_DICT = {}


@api(canonical_alias="nncf.torch.register_module")
def register_module(
    *quantizable_field_names: str, ignored_algorithms: list = None, target_weight_dim_for_compression: int = 0
):
    # quantizable_field_names will work for `weight` attributes only. Should later extend to registering
    # customly named attributes if it becomes necessary
    def wrap(cls):
        UNWRAPPED_USER_MODULES.registry_dict[cls.__name__] = cls
        nncf_wrapped_module_class_name = "NNCFUser{}".format(cls.__name__)
        NNCF_WRAPPED_USER_MODULES_DICT[cls] = type(nncf_wrapped_module_class_name, (_NNCFModuleMixin, cls), {})
        get_base_attributes_fn = lambda self: GenericWeightedLayerAttributes(
            self.weight.requires_grad, self.weight.shape
        )
        setattr(NNCF_WRAPPED_USER_MODULES_DICT[cls], "get_weight_shape", get_base_attributes_fn)
        if ignored_algorithms:
            setattr(NNCF_WRAPPED_USER_MODULES_DICT[cls], "ignored_algorithms", ignored_algorithms)

        setattr(
            NNCF_WRAPPED_USER_MODULES_DICT[cls], "target_weight_dim_for_compression", target_weight_dim_for_compression
        )
        return cls

    return wrap


def add_nncf_functionality_to_user_module(module: torch.nn.Module):
    user_class = module.__class__
    assert user_class.__name__ in UNWRAPPED_USER_MODULES.registry_dict
    module.__class__ = NNCF_WRAPPED_USER_MODULES_DICT[user_class]
    _NNCFModuleMixin.add_mixin_fields(module)
    return module


class RNNCellBaseNNCF(nn.Module):
    __constants__ = ["input_size", "hidden_size", "bias"]

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        linear_ih = nn.Linear(input_size, num_chunks * hidden_size, self.bias)
        linear_hh = nn.Linear(hidden_size, num_chunks * hidden_size, self.bias)
        self.weight_ih = linear_ih.weight
        self.weight_hh = linear_hh.weight
        self.bias_ih = linear_ih.bias
        self.bias_hh = linear_hh.bias
        self.linear_list = [linear_ih, linear_hh]
        self.reset_parameters()

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, input_):
        if input_.size(1) != self.input_size:
            raise nncf.ValidationError(
                "input_ has inconsistent input_size: got {}, expected {}".format(input_.size(1), self.input_size)
            )

    def check_forward_hidden(self, input_: torch.Tensor, hx: torch.Tensor, hidden_label: str = ""):
        if input_.size(0) != hx.size(0):
            raise nncf.ValidationError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input_.size(0), hidden_label, hx.size(0)
                )
            )

        if hx.size(1) != self.hidden_size:
            raise nncf.ValidationError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size
                )
            )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input_, hidden):
        raise NotImplementedError


ITERATION_MODULES = Registry("iteration_modules")


@ITERATION_MODULES.register()
class LSTMCellForwardNNCF(nn.Module):
    def __init__(self, input_linear, hidden_linear):
        super().__init__()
        self.input_linear = input_linear
        self.hidden_linear = hidden_linear

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hx, cx = hidden
        gates = self.input_linear(input_) + self.hidden_linear(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy


class LSTMCellNNCF(RNNCellBaseNNCF):
    def __init__(self, input_size=1, hidden_size=1, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)
        self.cell = LSTMCellForwardNNCF(self.linear_list[0], self.linear_list[1])

    def forward(self, input_, hidden=None):
        self.check_forward_input(input_)
        if hidden is None:
            zeros = torch.zeros(input_.size(0), self.hidden_size, dtype=input_.dtype, device=input_.device)
            hidden = (zeros, zeros)
        self.check_forward_hidden(input_, hidden[0], "[0]")
        self.check_forward_hidden(input_, hidden[1], "[1]")

        return self.cell(input_, hidden)


@ITERATION_MODULES.register()
class StackedRNN(nn.Module):
    class StackedRNNResetPoint(nn.Module):
        """
        Intentionally wrap concat, which is called inside nested loops, as a separate module.
        It allows not to add new node to nncf graph on each iteration of the loops.
        """

        def forward(self, all_output, input_):
            input_ = torch.cat(all_output, input_.dim() - 1)
            return input_

    def __init__(self, inners, num_layers, lstm=False, dropout=0):
        super().__init__()
        self.lstm = lstm
        self.num_layers = num_layers
        self.num_directions = int(len(inners) / num_layers)
        self.inners = nn.ModuleList(inners)
        self.total_layers = self.num_layers * self.num_directions
        self.dropout = dropout

    def forward(self, input_, hidden, batch_sizes):
        next_hidden = []

        if self.lstm:
            hidden = list(zip(*hidden))

        for i in range(self.num_layers):
            all_output = []
            for j in range(self.num_directions):
                k = i * self.num_directions + j
                hy, output = self.inners[k](input_, hidden[k], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input_ = self.StackedRNNResetPoint()(all_output, input_)
            if self.dropout != 0 and i < self.num_layers - 1:
                input_ = F.dropout(input_, p=self.dropout, training=self.training, inplace=False)

        if self.lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(self.total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(self.total_layers, *next_c[0].size()),
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(self.total_layers, *next_hidden[0].size())
        return next_hidden, input_


@ITERATION_MODULES.register()
class Recurrent(nn.Module):
    def __init__(self, cell, reverse=False):
        super().__init__()
        self.reverse = reverse
        self.cell = cell

    def forward(self, input_, hidden, batch_sizes=None):
        output = []
        steps = range(input_.size(0) - 1, -1, -1) if self.reverse else range(input_.size(0))
        for i in steps:
            with forward_nncf_trace():
                hidden_input = input_[i]
            hidden = self.cell(hidden_input, hidden)
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if self.reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input_.size(0), *output[0].size())

        return hidden, output


def variable_recurrent_factory():
    def factory(cell, reverse=False):
        if reverse:
            return VariableRecurrentReverse(cell)
        return VariableRecurrent(cell)

    return factory


@ITERATION_MODULES.register()
class VariableRecurrent(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input_, hidden, batch_sizes):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        with forward_nncf_trace():
            batch_size_elements = [b for b in batch_sizes]
        for batch_size in batch_size_elements:
            step_input = input_[input_offset : input_offset + batch_size]
            input_offset += batch_size

            bs_decrease = last_batch_size - batch_size
            if bs_decrease > 0:
                hidden_len = len(hidden)
                hidden_offset_elts = []
                hidden_offset_elts_reversed = []
                for i in range(hidden_len):
                    with forward_nncf_trace():
                        hidden_offset_elts.append(hidden[i][-bs_decrease:])
                        hidden_offset_elts_reversed.append(hidden[i][:-bs_decrease])

                hiddens.append(tuple(hidden_offset_elts))
                hidden = tuple(hidden_offset_elts_reversed)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (self.cell(step_input, hidden[0]),)
            else:
                hidden = self.cell(step_input, hidden)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)
        return hidden, output


@ITERATION_MODULES.register()
class VariableRecurrentReverse(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input_, hidden, batch_sizes):
        output = []
        input_offset = input_.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[: batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            hidden = self.ReverseResetPoint()(batch_size, hidden, inc, initial_hidden, last_batch_size)
            last_batch_size = batch_size
            step_input = input_[input_offset - batch_size : input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (self.cell(step_input, hidden[0]),)
            else:
                hidden = self.cell(step_input, hidden)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    @ITERATION_MODULES.register()
    class ReverseResetPoint(nn.Module):
        """
        Intentionally wrap concat undef if condition as a separate module
        to prevent adding new node to nncf graph on each iteration
        """

        def forward(self, batch_size, hidden, inc, initial_hidden, last_batch_size):
            if inc > 0:
                hidden = tuple(
                    torch.cat((h, ih[last_batch_size:batch_size]), 0) for h, ih in zip(hidden, initial_hidden)
                )
            return hidden


class NNCF_RNN(nn.Module):
    """Common class for RNN modules. Currently, LSTM is supported only"""

    def __init__(
        self,
        mode="LSTM",
        input_size=1,
        hidden_size=1,
        num_layers=1,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        bias=True,
    ):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            nncf_logger.debug(
                f"dropout option adds dropout after all but last recurrent layer, "
                f"so non-zero dropout expects num_layers greater than 1, "
                f"but got dropout={dropout} and num_layers={num_layers}"
            )

        if mode == "LSTM":
            gate_size = 4 * hidden_size
            self.cell_type = LSTMCellForwardNNCF
        else:
            # elif mode == 'GRU':
            #     gate_size = 3 * hidden_size
            # elif mode == 'RNN_TANH':
            #     gate_size = hidden_size
            # elif mode == 'RNN_RELU':
            #     gate_size = hidden_size
            # else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._all_weights = []
        self.cells = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                linear_ih = nn.Linear(layer_input_size, gate_size, bias)
                linear_hh = nn.Linear(hidden_size, gate_size, bias)
                self.cells.append(self.cell_type(linear_ih, linear_hh))
                params = (linear_ih.weight, linear_hh.weight, linear_ih.bias, linear_hh.bias)
                suffix = "_reverse" if direction == 1 else ""
                weight_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    weight_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                weight_names = [x.format(layer, suffix) for x in weight_names]
                for name, param in zip(weight_names, params):
                    setattr(self, name, param)
                self._all_weights.append(weight_names)

        self.reset_parameters()
        self.variable_length = True
        self.rnn_impl = self.get_rnn_impl(self.variable_length, self.cells)

    def get_rnn_impl(self, variable_length, cells):
        if variable_length:
            rec_factory = variable_recurrent_factory()
        else:
            rec_factory = Recurrent
        inners = []
        for layer_idx in range(self.num_layers):
            idx = layer_idx * self.num_directions
            if self.bidirectional:
                layer_inners = [rec_factory(cells[idx]), rec_factory(cells[idx + 1], reverse=True)]
            else:
                layer_inners = [
                    rec_factory(cells[idx]),
                ]
            inners.extend(layer_inners)
        return StackedRNN(inners, self.num_layers, (self.mode == "LSTM"), dropout=self.dropout)

    def check_forward_args(self, input_, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input_.dim() != expected_input_dim:
            raise nncf.ValidationError(
                "input_ must have {} dimensions, got {}".format(expected_input_dim, input_.dim())
            )
        if self.input_size != input_.size(-1):
            raise nncf.ValidationError(
                "input_.size(-1) must be equal to input_size. Expected {}, got {}".format(
                    self.input_size, input_.size(-1)
                )
            )

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input_.size(0) if self.batch_first else input_.size(1)

        expected_hidden_size = (mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg="Expected hidden size {}, got {}"):
            expected_size = self.num_layers * self.num_directions
            if expected_size != len(hx):
                raise nncf.InternalError("Expected number of hidden states {}, got {}".format(expected_size, len(hx)))
            for element in hx:
                if tuple(element.size()) != expected_hidden_size:
                    raise nncf.InternalError(msg.format(expected_hidden_size, tuple(element.size())))

        if self.mode == "LSTM":
            check_hidden_size(hidden[0], expected_hidden_size, "Expected hidden[0] size {}, got {}")
            check_hidden_size(hidden[1], expected_hidden_size, "Expected hidden[1] size {}, got {}")
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    @staticmethod
    def apply_permutation(tensor: torch.Tensor, permutation: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return tensor.index_select(dim, permutation)

    def permute_hidden(
        self, hx: torch.Tensor, permutation: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if permutation is None:
            return hx
        return self.apply_permutation(hx[0], permutation), self.apply_permutation(hx[1], permutation)

    def prepare_hidden(
        self, hx: torch.Tensor, permutation: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]]:
        if permutation is None:
            return hx
        split_size = len(hx[0])
        concat_hx = torch.cat([torch.unsqueeze(t, 0) for t in hx[0]])
        concat_cx = torch.cat([torch.unsqueeze(t, 0) for t in hx[1]])
        permuted_hidden = self.apply_permutation(concat_hx, permutation), self.apply_permutation(concat_cx, permutation)
        hc = permuted_hidden[0].chunk(split_size, 0)
        cc = permuted_hidden[1].chunk(split_size, 0)
        hidden = (tuple(torch.squeeze(c, 0) for c in hc), tuple(torch.squeeze(c, 0) for c in cc))
        return hidden

    def forward(self, input_, hidden=None):
        is_packed = isinstance(input_, PackedSequence)

        sorted_indices = None
        unsorted_indices = None
        if is_packed:
            input_, batch_sizes, sorted_indices, unsorted_indices = input_
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input_.size(0) if self.batch_first else input_.size(1)

        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            hidden = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                requires_grad=False,
                device=input_.device,
            )
            if self.mode == "LSTM":
                hidden = (hidden, hidden)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hidden = self.prepare_hidden(hidden, sorted_indices)

        self.check_forward_args(input_, hidden, batch_sizes)

        is_currently_variable = batch_sizes is not None
        if self.variable_length and not is_currently_variable or not self.variable_length and is_currently_variable:
            # override rnn_impl, it's assumed that this should happen very seldom, as
            # usually there's only one mode active whether variable length, or constant ones
            self.rnn_impl = self.get_rnn_impl(is_currently_variable, self.cells)

        if self.batch_first and batch_sizes is None:
            input_ = input_.transpose(0, 1)

        hidden, output = self.rnn_impl(input_, hidden, batch_sizes)

        if self.batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)

        return output, self.permute_hidden(hidden, unsorted_indices)
