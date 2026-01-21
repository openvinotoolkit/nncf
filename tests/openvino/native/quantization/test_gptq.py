# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License
# Copyright (c) 2023 潘其威(William)

import math

import numpy as np
import openvino as ov
import pytest
import torch

from nncf import Dataset
from nncf.common.factory import build_graph
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.gptq import GPTQ
from nncf.tensor.definitions import TensorDataType
from nncf.tensor.tensor import Tensor

# Modification Notes:
# This unit test is adapted from:
# https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/quantization/quantizer.py


def quantize(x, scale, zero, minq, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return scale * (q - zero)


class GPTQQuantizer(torch.nn.Module):
    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            if self.sym:
                self.scale = torch.where(-xmin > xmax, -xmin, -xmax)
                self.scale = self.scale / ((self.maxq + 1) / 2)
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / self.maxq
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            minq = 0
            maxq = self.maxq
            if self.sym:
                minq = -(self.maxq + 1) / 2
                maxq = (self.maxq + 1) / 2 - 1
            return quantize(x, self.scale, self.zero, minq, maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQReference:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, torch.nn.Conv2d):
            W = W.flatten(1)
        # if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
        #     W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = GPTQQuantizer()
        self.quantizer.configure(4, perchannel=True, sym=True)

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # if isinstance(self.layer, (torch.nn.Linear, transformers.Conv1D)):
        if isinstance(self.layer, (torch.nn.Linear)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, torch.nn.Conv2d):
            unfold = torch.nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=False,
        static_groups=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, torch.nn.Conv2d):
            W = W.flatten(1)
        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + group_size)], weight=True)

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        # if isinstance(self.layer, transformers.Conv1D):
        #     Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx


class Linear3DModel(torch.nn.Module):
    def __init__(self, weight_3d: np.ndarray):
        super().__init__()
        w = torch.from_numpy(weight_3d)
        self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        # OV expects transposed constant when applying GPTQ
        return torch.matmul(x, self.weight.transpose(1, 2))


def _create_ov_model(weights: np.ndarray, input_shape: tuple, is_3d_weights: bool = False):
    import openvino.runtime.opset13 as opset

    param = opset.parameter(input_shape, dtype=np.float32, name="input")
    const = opset.constant(weights, dtype=np.float32, name="self.weight")
    matmul = opset.matmul(param, const, transpose_a=False, transpose_b=True)
    result = opset.result(matmul, name="output")
    return ov.Model([result], [param])


def _make_nncf_dataset(ov_model, inputs: list[np.ndarray]) -> Dataset:
    input_name = ov_model.inputs[0].get_any_name()
    items = [{input_name: inp} for inp in inputs]
    return Dataset(items, lambda x: x)


@pytest.mark.parametrize("is_3d_weights", [False, True], ids=["2D_weights", "3D_weights"])
def test_calculate_scale_linear(is_3d_weights: bool):
    np.random.seed(0)

    hidden_dims = 32
    out_dims = 20
    group_size = 16
    n_inputs = 10
    batch_size = 4

    inputs = [np.random.rand(batch_size, 128, hidden_dims).astype(np.float32) for _ in range(n_inputs)]
    weights = np.random.rand(batch_size, out_dims, hidden_dims).astype(np.float32)

    # Select only first batch for 2D case and make it 2D
    inputs = inputs if is_3d_weights else [activation[0] for activation in inputs]
    weights = weights if is_3d_weights else weights[0]

    # Step 1: We apply a reference GPTQ implementation on the Pytorch model. This gives us a ground truth of scales and
    # Quantized values
    with torch.no_grad():
        weight = weights if is_3d_weights else np.expand_dims(weights, axis=0)
        # Inputs for 3D is a list of shape [n_inputs, batch_size, 128, in_dim]
        # For 2D it is [n_inputs, 1, 128, in_dim]. We unsqueeze at 0 for every input tensor
        batched_inputs = inputs if is_3d_weights else [np.expand_dims(x, axis=0) for x in inputs]

        pt_model = Linear3DModel(weights) if is_3d_weights else torch.nn.Linear(hidden_dims, out_dims, bias=False)
        if not is_3d_weights:
            pt_model.weight.copy_(torch.from_numpy(weights))

        ref_gptqs, ref_scales = [], []
        for batch in range(weight.shape[0]):
            layer = torch.nn.Linear(hidden_dims, out_dims, bias=False)
            layer.weight.copy_(torch.from_numpy(weight[batch]))

            ref_gptq = GPTQReference(layer)
            for inp in batched_inputs:
                ref_gptq.add_batch(torch.from_numpy(inp[batch]))

            ref_scale_for_batch, _, _ = ref_gptq.fasterquant(percdamp=0.1, group_size=group_size)
            ref_gptqs.append(ref_gptq)
            ref_scales.append(ref_scale_for_batch)

        ref_scale = np.stack([s.detach().cpu().numpy() for s in ref_scales], axis=0)

    # Step 2: Create OV models so that we can use nncf to compress this with our own GPTQ
    # We do not use ov.convert_model() here since we expect a specific transposed weight and
    # not transposed activation. It is hard to create and convert such a model from PT -> OV
    # due to things like constant folding etc. which are automatically performed or generally
    # hard to translate.
    ov_model = _create_ov_model(weights, inputs[0].shape, is_3d_weights)
    graph = build_graph(ov_model)

    # Step 3: Setup and apply GPTQ as usual
    gptq = GPTQ()
    gptq._set_backend_entity(ov_model)

    node_with_weight = graph.get_all_nodes()[1]

    wrapped_inputs = [Tensor(inp) for inp in inputs]
    H = gptq._calculate_hessian(node_with_weight, wrapped_inputs, is_3d_weight=is_3d_weights)

    reference_gptq_list = ref_gptqs if is_3d_weights else [ref_gptq]
    nncf_hessian_list = H.data if is_3d_weights else np.expand_dims(H.data, axis=0)

    for batch_index, (reference_gptq, nncf_hessian) in enumerate(zip(reference_gptq_list, nncf_hessian_list)):
        reference_hessian = reference_gptq.H.detach().numpy()
        assert np.all(np.isclose(reference_hessian, nncf_hessian)), f"Hessian mismatch for batch {batch_index}"

    nncf_dataset = _make_nncf_dataset(ov_model, inputs)
    reduction_axes = (1,) if not is_3d_weights else (2,)
    wc_param = WeightCompressionParameters(
        weight_name="self.weight",
        node_with_weight=node_with_weight,
        weight_port_id=1,
        weight_dtype=TensorDataType.float32,
        weight_shape=weights.shape,
        reduction_axes=reduction_axes,
    )
    wc_param.compression_config = WeightCompressionConfig(mode=CompressWeightsMode.INT4_SYM, group_size=group_size)
    wc_params = [wc_param]

    _, res = gptq.apply(ov_model, graph, nncf_dataset, wc_params)

    # Step 4: Obtain the scales from our GPTQ implementation and compare with reference
    scale_from_nncf = res.get("self.weight").scale.data
    ref_scale = ref_scale.numpy() if isinstance(ref_scale, torch.Tensor) else ref_scale
    ref_scale = ref_scale.reshape(scale_from_nncf.shape)

    assert np.all(np.isclose(ref_scale, scale_from_nncf))
    # Here we obtain weights from the model instead of apply directly so that we also check
    # if the weight is changed in the OV model.
    ov_weight = gptq._backend_entity.get_weight(node_with_weight, 1, ov_model, graph)
    ref_weights = np.stack(
        [ref_gptq.layer.weight.detach().numpy() for ref_gptq in ref_gptqs],
        axis=0,
    )

    assert np.all(np.isclose(ref_weights, ov_weight.data))
