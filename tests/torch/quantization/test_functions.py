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
from typing import Callable, List, Optional

import numpy as np
import pytest
import torch
from torch.autograd import Variable
from torch.distributions.uniform import Uniform

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.quantize_functions import asymmetric_quantize
from nncf.torch.quantization.quantize_functions import get_scale_zp_from_input_low_input_high
from nncf.torch.quantization.quantize_functions import symmetric_quantize
from nncf.torch.quantization.reference import ReferenceBackendType
from nncf.torch.quantization.reference import ReferenceQuantize
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import get_grads

EPS = 1e-6


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def idfn(val):
    if isinstance(val, list):
        return "[{}]".format("-".join([str(v) for v in val]))

    return None


RQ = ReferenceQuantize(backend_type=ReferenceBackendType.NUMPY)


def generate_one_channel_input(
    input_low: np.ndarray,
    input_range: np.ndarray,
    ch_idx: Optional[List[int]],
    input_size: List[int],
    bits: int,
    get_deviation: Callable[[], int],
    min_deviation: float,
):
    if np.prod(input_low.shape) == 1:
        input_low_ch, input_range_ch = map(lambda x: x.squeeze(), [input_low, input_range])
    else:
        input_low_ch, input_range_ch = map(lambda x: x[tuple(ch_idx)].squeeze(), [input_low, input_range])

    quant_len_ch = input_range_ch / (2**bits - 1)
    points = np.array([input_low_ch + (i // 2 + get_deviation()) * quant_len_ch for i in range(2 ** (bits + 1) - 2)])
    input_elems = np.prod(input_size)
    if input_elems > len(points):
        points = np.tile(points, int(np.ceil(input_elems / len(points) + 0.5)))
    points = points[:input_elems]
    out_range_fraction = 0.5
    out_range_elems = int(input_elems * out_range_fraction)
    points[:out_range_elems] = np.hstack(
        [
            input_low_ch - np.random.random_sample(out_range_elems // 2 + out_range_elems % 2) - min_deviation,
            input_low_ch + input_range_ch + np.random.random_sample(out_range_elems // 2) + min_deviation,
        ]
    )
    return np.random.permutation(points).reshape(input_size)


def generate_input(
    input_size: List[int],
    input_low: np.ndarray,
    input_range: np.ndarray,
    bits: int,
    scale_mode: str,
    is_weights: bool,
    middle_points: bool = False,
    min_deviation: float = 0.0,
    max_deviation: float = 0.4,
):
    def assert_input_size(input_size):
        assert np.prod(input_size) >= 4, "amount of input elements is to low to cover all corner cases"

    if abs(min_deviation - max_deviation) < 1e-5:

        def random_deviation():
            return min_deviation

    else:

        def random_deviation():
            return min_deviation + np.random.random_sample() * (max_deviation - min_deviation)

    if middle_points:

        def get_deviation():
            get_deviation.state += 1
            if get_deviation.state % 2:
                return 0.5
            return random_deviation()

        get_deviation.state = 0
    else:

        def get_deviation():
            return random_deviation()

    if scale_mode == "single_scale":
        assert_input_size(input_size)
        return generate_one_channel_input(input_low, input_range, None, input_size, bits, get_deviation, min_deviation)
    inputs = None
    if scale_mode == "per_channel_scale":
        if is_weights:
            assert_input_size(input_size[1:])
            channel_count = input_size[0]
            if channel_count == 1:
                pytest.skip("Same case as for single scale mode")
            ch_idx = [0 for _ in input_size]
            inputs = np.empty(input_size)
            for idx in range(0, channel_count):
                ch_idx[0] = idx
                inputs[idx] = generate_one_channel_input(
                    input_low, input_range, ch_idx, input_size[1:], bits, get_deviation, min_deviation
                )
        else:
            assert_input_size(input_size[0:1] + input_size[2:])
            channel_count = input_size[1]
            if channel_count == 1:
                pytest.skip("Same case as for single scale mode")
            ch_idx = [0 for _ in input_size]
            inputs = np.empty(input_size)
            for idx in range(0, channel_count):
                ch_idx[1] = idx
                inputs[:, idx] = generate_one_channel_input(
                    input_low, input_range, ch_idx, input_size[0:1] + input_size[2:], bits, get_deviation, min_deviation
                )
    return inputs


def get_test_data(data_list, is_cuda=False, is_backward=False, is_fp16=False):
    results = []
    for data in data_list:
        result = torch.from_numpy(data.copy())
        if is_cuda:
            result = result.cuda()
        if is_fp16:
            result = result.half()
        else:
            result = result.float()
        if is_backward:
            result = Variable(result, requires_grad=True)
        results.append(result)
    return results if len(data_list) > 1 else results[0]


def skip_if_half_on_cpu(is_fp16, use_cuda):
    if is_fp16 and not use_cuda:
        pytest.skip(
            "As of PyTorch 1.5, the 'abs' operation is not supported on CPU for half and therefore"
            " symmetric quantize fails. Remove this once this is fixed in PyTorch."
        )


def check_quant_moved(test_input, test_val, ref_val, quant_len, input_low, input_range, is_fp16, rtol, atol=1e-10):
    """
    Checks values in `test_val` are inside of closest quant and
    values in `test_val` and `ref_val` elementwise either equal with given rtol/atol or
    values differ by correspondent `quant_len` +- rtol.

    :param test_input: Input of a quantizer.
    :param test_val: Given test value.
    :param ref_val: Given reference value.
    :param quant_len: Length of quants in quantizers
        (for each channel in case per channel quantization).
    :param atol: Absolute tolerance.
    :param rtol: Relative tolerance.
    """

    def to_tensor(a):
        return torch.tensor(a, dtype=test_input.dtype, device=test_input.device)

    mask_in = (to_tensor(input_low) < test_input).logical_and(test_input < to_tensor(input_low + input_range))
    quant_len_broadcasted = torch.masked_select(to_tensor(quant_len), mask_in)
    # Due to FP16 arithmetic difference could be equal to
    # one quant. 0.05 is added as an eps diff between two fp16 values
    inp_out_deviation_mult = 1.05 if is_fp16 else 1.0
    inp_out_abs_diff = (test_input[mask_in] - test_val[mask_in]).abs()
    assert (
        inp_out_abs_diff < inp_out_deviation_mult * quant_len_broadcasted
    ).all(), "quantized values are outside of closest quant"

    t_numpy = test_val.cpu().detach().numpy()
    bad_elems = ~np.isclose(t_numpy, ref_val, rtol, atol)
    if not np.any(bad_elems):
        return

    abs_diff = np.abs(t_numpy - ref_val)
    if np.prod(quant_len.shape) > 1:
        ch_dim = np.argmax(quant_len.shape)
        idxs = np.transpose(np.where(bad_elems))
        quant_len = quant_len.squeeze()[idxs[:, ch_dim]]
    moved_quant_elems = np.abs(abs_diff[bad_elems] - quant_len) < rtol
    assert np.all(moved_quant_elems)


def check_outputs_for_quantization_functions(test_val: torch.Tensor, ref_val: np.ndarray, rtol=1e-4, atol=1e-10):
    PTTensorListComparator.check_equal(test_val, ref_val, rtol, atol)


@pytest.mark.parametrize("bits", (8, 4), ids=("8bit", "4bit"))
@pytest.mark.parametrize("scale_mode", ["single_scale", "per_channel_scale"])
@pytest.mark.parametrize("is_fp16", (True, False), ids=("fp16", "fp32"))
class BaseParametrized:
    class TestSymmetric:
        @staticmethod
        def generate_scale(input_size, scale_mode, is_weights, is_fp16, fixed=None):
            assert scale_mode in ["single_scale", "per_channel_scale"]

            if fixed is not None:

                def calc_scale():
                    return fixed

            else:

                def calc_scale():
                    min_scale = 1.0 if is_fp16 else 0.1
                    return min_scale + np.random.random_sample((1,)) * (1 - min_scale)

            if scale_mode == "single_scale":
                return np.array([calc_scale()])

            if scale_mode == "per_channel_scale":
                if is_weights:
                    channel_count = input_size[0]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_size]
                    scales_shape[0] = channel_count
                    scales = np.empty(scales_shape)
                    for idx in range(0, channel_count):
                        scales[idx] = calc_scale()
                else:
                    channel_count = input_size[1]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_size]
                    scales_shape[1] = channel_count
                    scales = np.empty(scales_shape)
                    for idx in range(0, channel_count):
                        scales[0, idx] = calc_scale()
            return scales

        @staticmethod
        def get_range_level(is_signed, bits):
            levels = 2**bits
            if is_signed:
                level_high = (levels // 2) - 1
                level_low = -(levels // 2)
            else:
                level_high = levels - 1
                level_low = 0
            return level_low, level_high, levels

        def test_quantize_symmetric_forward(
            self, _seed, is_signed, is_weights, is_fp16, input_size, bits, use_cuda, scale_mode
        ):
            if not torch.cuda.is_available() and use_cuda is True:
                pytest.skip("Skipping CUDA test cases for CPU only setups")
            skip_if_half_on_cpu(is_fp16, use_cuda)
            if is_fp16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32

            ref_scale = self.generate_scale(input_size, scale_mode, is_weights, is_fp16).astype(np_dtype)
            test_scale = get_test_data([ref_scale], use_cuda, is_fp16=is_fp16)

            ref_scale = abs(ref_scale) + EPS
            level_low, level_high, levels = self.get_range_level(is_signed, bits)

            ref_input_low = ref_scale * (level_low / level_high)
            ref_input_range = ref_scale - ref_input_low

            ref_input = generate_input(
                input_size, ref_input_low, ref_input_range, bits, scale_mode, is_weights, middle_points=True
            ).astype(np_dtype)
            test_input = get_test_data([ref_input], use_cuda, is_fp16=is_fp16)

            for array_ in (ref_input, ref_input_low, ref_input_range):
                assert array_.dtype == np_dtype
            for tensor_ in (test_input, test_scale):
                assert tensor_.dtype == torch.half if is_fp16 else torch.float

            ref_value = RQ.forward(ref_input, ref_input_low, ref_input_range, levels)
            test_value = symmetric_quantize(test_input, levels, level_low, level_high, test_scale, EPS)
            if use_cuda:
                quant_len = ref_input_range / (2**bits - 1)
                check_quant_moved(
                    test_input,
                    test_value,
                    ref_value,
                    quant_len,
                    ref_input_low,
                    ref_input_range,
                    is_fp16,
                    rtol=1e-2 if is_fp16 else 1e-3,
                )
            else:
                # Note: this will fail in case CPU reference quantization implementations had to be used instead of the
                # compiled extensions, since they don't work well with middle-of-quanta values.
                check_outputs_for_quantization_functions(test_value, ref_value, rtol=1e-2 if is_fp16 else 1e-3)

        def test_quantize_symmetric_backward(
            self, _seed, is_signed, is_weights, is_fp16, input_size, bits, use_cuda, scale_mode
        ):
            if not torch.cuda.is_available() and use_cuda is True:
                pytest.skip("Skipping CUDA test cases for CPU only setups")
            skip_if_half_on_cpu(is_fp16, use_cuda)
            if is_fp16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32

            fixed = None
            if is_fp16:
                # This is needed to make scale == 1 to prevent
                # quant movement on forward pass in FP16 precision.
                # In case scale != 1., not precice scale multiplication in FP16
                # could lead to big deviations, so even if an input point
                # lies in safe range (far from middles of quants) after a scaling
                # it could end up in the middle of a quant. It happens mostly
                # when target quant > 150 because in real life scenarious quantization range
                # usualy less than 2 ** quantization bits,
                # so input is small and scale is big, small FP16 input multiplies big fp16 scale,
                # deviation is significant.
                fixed = 2 ** (bits - 1) - 1
            ref_scale = self.generate_scale(input_size, scale_mode, is_weights, is_fp16, fixed=fixed).astype(np_dtype)
            test_scale = get_test_data([ref_scale], use_cuda, is_backward=True, is_fp16=is_fp16)
            level_low, level_high, levels = self.get_range_level(is_signed, bits)

            ref_scale = abs(ref_scale) + EPS

            ref_input_low = ref_scale * (level_low / level_high)
            ref_input_range = ref_scale - ref_input_low

            # This is needed to prevent middle points in forward pass.
            # Small deviation in computations could lead to completely different
            # results: instead of an n-th quantum the n+1-st quant could be returned.
            # Different results in forward leads to high deviations on backward pass.
            # To prevent this, input values are put into [min_deviation, max_deviation]
            # section of quantum, so small deviation won't change the quantum on forward pass
            min_deviation = 0.1 if is_fp16 else 0.0
            max_deviation = 0.35 if is_fp16 else 0.4
            ref_input = generate_input(
                input_size,
                ref_input_low,
                ref_input_range,
                bits,
                scale_mode,
                is_weights,
                not use_cuda,
                min_deviation,
                max_deviation,
            ).astype(np_dtype)
            test_input = get_test_data([ref_input], use_cuda, is_backward=True, is_fp16=is_fp16)

            for array_ in (ref_input, ref_input_low, ref_input_range):
                assert array_.dtype == np_dtype
            for tensor_ in (test_input, test_scale):
                assert tensor_.dtype == torch.half if is_fp16 else torch.float

            ref_output = RQ.forward(ref_input, ref_input_low, ref_input_range, levels)

            mock_prev_output_grads = np.ones(input_size, dtype=np.float16 if is_fp16 else np.float32)
            ref_grads = RQ.backward(
                mock_prev_output_grads, ref_input, ref_input_low, ref_input_range, ref_output, level_low, level_high
            )
            del ref_grads[1]
            test_value = symmetric_quantize(test_input, levels, level_low, level_high, test_scale, EPS)
            test_value.sum().backward()
            test_grads = get_grads([test_input, test_scale])

            check_outputs_for_quantization_functions(test_value, ref_output, rtol=1e-2 if is_fp16 else 1e-3)
            check_outputs_for_quantization_functions(test_grads, ref_grads, rtol=1e-2 if is_fp16 else 1e-3)

    class TestAsymmetric:
        @classmethod
        def generate_range(cls, input_size, scale_mode, is_weights, is_fp16, levels, fixed=None):
            np_dtype = np.float16 if is_fp16 else np.float32
            return map(
                lambda x: x.astype(np_dtype),
                cls.generate_range_fp64(input_size, scale_mode, is_weights, fixed, is_fp16, levels),
            )

        @staticmethod
        def generate_range_fp64(input_size, scale_mode, is_weights, fixed, is_fp16, levels):
            assert scale_mode in ["single_scale", "per_channel_scale"]

            if fixed is not None:

                def calc_low_and_range():
                    return fixed["input_low"], fixed["input_range"]

            else:

                def calc_low_and_range():
                    min_range = 1.0 if is_fp16 else 0.1
                    input_low = np.random.random_sample() * 3 - 1.5
                    input_range = min_range + np.random.random_sample() * 3
                    if levels <= 16 and is_fp16:
                        scale = (levels - 1) / input_range
                        zp_fp = -input_low * scale
                        # Checks zp_fp is not near the middle of two integer values
                        # and fix it in case it is.
                        if np.abs(np.abs(zp_fp - np.round(zp_fp)) - 0.5) < 0.1:
                            # Make zp_fp += 0.2
                            new_input_low = input_low + 0.2 / scale
                            # Preserve sing of the input_low
                            if np.sign(new_input_low) != np.sign(input_low):
                                input_low -= 0.2 / scale
                            else:
                                input_low = new_input_low
                    return input_low, input_range

            if scale_mode == "single_scale":
                input_low, input_range = calc_low_and_range()
                return np.array([input_low]), np.array([input_range])

            if scale_mode == "per_channel_scale":
                if is_weights:
                    channel_count = input_size[0]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_size]
                    scales_shape[0] = channel_count
                    input_low = np.empty(scales_shape)
                    input_range = np.empty(scales_shape)
                    for idx in range(0, channel_count):
                        input_low[idx], input_range[idx] = calc_low_and_range()
                else:
                    channel_count = input_size[1]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_size]
                    scales_shape[1] = channel_count
                    input_low = np.empty(scales_shape)
                    input_range = np.empty(scales_shape)
                    for idx in range(0, channel_count):
                        input_low[0, idx], input_range[0, idx] = calc_low_and_range()

            return input_low, input_range

        @staticmethod
        def get_range_level(bits):
            levels = 2**bits
            level_low = 0
            level_high = levels - 1
            return level_low, level_high, levels

        def test_quantize_asymmetric_forward(
            self, request, _seed, input_size, bits, use_cuda, is_weights, is_fp16, scale_mode
        ):
            if not torch.cuda.is_available() and use_cuda is True:
                pytest.skip("Skipping CUDA test cases for CPU only setups")

            skip_if_half_on_cpu(is_fp16, use_cuda)
            if is_fp16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32

            level_low, level_high, levels = self.get_range_level(bits)
            ref_input_low, ref_input_range = self.generate_range(input_size, scale_mode, is_weights, is_fp16, levels)
            test_input_low, test_input_range = get_test_data(
                [ref_input_low, ref_input_range], use_cuda, is_fp16=is_fp16
            )

            ref_input_range = abs(ref_input_range) + EPS
            ref_input_low, ref_input_range = RQ.tune_range(ref_input_low, ref_input_range, levels)

            ref_input = generate_input(
                input_size, ref_input_low, ref_input_range, bits, scale_mode, is_weights, True
            ).astype(np_dtype)
            test_input = get_test_data([ref_input], use_cuda, is_fp16=is_fp16)

            for array_ in (ref_input, ref_input_low, ref_input_range):
                assert array_.dtype == np_dtype
            for tensor_ in (test_input, test_input_low, test_input_range):
                assert tensor_.dtype == torch.half if is_fp16 else torch.float

            ref_value = RQ.forward(ref_input, ref_input_low, ref_input_range, levels)
            test_value = asymmetric_quantize(
                test_input, levels, level_low, level_high, test_input_low, test_input_range, EPS
            )

            if use_cuda:
                quant_len = ref_input_range / (2**bits - 1)
                check_quant_moved(
                    test_input,
                    test_value,
                    ref_value,
                    quant_len,
                    ref_input_low,
                    ref_input_range,
                    is_fp16,
                    rtol=1e-2 if is_fp16 else 1e-3,
                )
            else:
                # Note: this will fail in case CPU reference quantization implementations had to be used instead of the
                # compiled extensions, since they don't work well with middle-of-quanta values.
                check_outputs_for_quantization_functions(test_value, ref_value, rtol=1e-2 if is_fp16 else 1e-3)

        def test_quantize_asymmetric_backward(self, _seed, input_size, bits, use_cuda, is_weights, is_fp16, scale_mode):
            if not torch.cuda.is_available() and use_cuda is True:
                pytest.skip("Skipping CUDA test cases for CPU only setups")
            skip_if_half_on_cpu(is_fp16, use_cuda)
            if is_fp16:
                np_dtype = np.float16
            else:
                np_dtype = np.float32
            level_low, level_high, levels = self.get_range_level(bits)
            fixed = None
            if is_fp16:
                # This is needed to make scale == 1 to prevent
                # quant movement on forward pass in FP16 precision.
                # In case scale != 1., not precise scale multiplication in FP16
                # could lead to big deviations, so even if an input point
                # lies in safe range (far from middles of quants) after a scaling
                # it could end up in the middle of a quant. It happens mostly
                # when target quant > 150 because in real life scenarios quantization range
                # usually less than 2 ** quantization bits,
                # so input is small and scale is big, small FP16 input multiplies big fp16 scale,
                # deviation is significant.
                fixed = {}
                fixed["input_low"] = -(2 ** (bits - 1))
                fixed["input_range"] = 2**bits - 1
            ref_input_low, ref_input_range = self.generate_range(
                input_size, scale_mode, is_weights, is_fp16, levels, fixed
            )
            test_input_low, test_input_range = get_test_data(
                [ref_input_low, ref_input_range], use_cuda, is_backward=True, is_fp16=is_fp16
            )

            ref_input_range = abs(ref_input_range) + EPS
            ref_input_low, ref_input_range = RQ.tune_range(ref_input_low, ref_input_range, levels)

            for tensor_ in (test_input_low, test_input_range):
                assert tensor_.dtype == torch.half if is_fp16 else torch.float

            # This is needed to prevent middle points in forward pass.
            # Small deviation in computations could lead to completely different
            # results: instead of quant a quant a + 1 will be returned.
            # Different results in forward leads to high deviations on backward pass.
            # To prevent this, input values are put into [min_deviation, max_deviation]
            # section of quant, so small deviation won't change the quant on forward pass
            min_deviation = 0.1 if is_fp16 else 0.0
            max_deviation = 0.35 if is_fp16 else 0.4
            ref_input = generate_input(
                input_size,
                ref_input_low,
                ref_input_range,
                bits,
                scale_mode,
                is_weights,
                not use_cuda,
                min_deviation,
                max_deviation,
            ).astype(np_dtype)
            test_input = get_test_data([ref_input], use_cuda, is_fp16=is_fp16, is_backward=True)

            for array_ in (ref_input, ref_input_low, ref_input_range):
                assert array_.dtype == np_dtype
            for tensor_ in (test_input, test_input_low, test_input_range):
                assert tensor_.dtype == torch.half if is_fp16 else torch.float

            ref_output = RQ.forward(ref_input, ref_input_low, ref_input_range, levels)

            mock_prev_output_grads = np.ones(input_size, dtype=np.float16 if is_fp16 else np.float32)
            ref_grads = RQ.backward(
                mock_prev_output_grads, ref_input, ref_input_low, ref_input_range, ref_output, level_low, level_high
            )

            test_value = asymmetric_quantize(
                test_input, levels, level_low, level_high, test_input_low, test_input_range, eps=EPS
            )
            test_value.sum().backward()
            test_grads = get_grads([test_input, test_input_low, test_input_range])

            check_outputs_for_quantization_functions(test_value, ref_output, rtol=1e-2 if is_fp16 else 1e-3)

            check_outputs_for_quantization_functions(test_grads, ref_grads, rtol=1e-2 if is_fp16 else 1e-3)


@pytest.mark.parametrize("input_size", [[1, 16, 64, 64], [4, 16, 16, 16]], ids=idfn)
class TestParametrizedFast(BaseParametrized):
    pass


@pytest.mark.nightly
@pytest.mark.parametrize(
    "input_size",
    [[1, 48, 112, 112], [1, 96, 28, 28], [1, 288, 14, 14], [16, 96, 112, 112], [16, 192, 28, 28], [16, 576, 14, 14]],
    ids=idfn,
)
class TestParametrizedLong(BaseParametrized):
    pass


def test_mapping_to_zero(use_cuda, quantization_mode):
    torch.manual_seed(42)

    if use_cuda and not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    device = "cuda" if use_cuda else "cpu"
    x_zero = torch.zeros([1]).to(torch.device(device))
    levels = 256
    eps = 1e-6
    number_of_samples = 100

    if quantization_mode == QuantizationMode.SYMMETRIC:
        level_low = -128
        level_high = 127

        uniform_dist_scale = Uniform(0, 100)
        for _ in range(number_of_samples):
            scale = uniform_dist_scale.sample().to(torch.device(device))
            test_output = symmetric_quantize(x_zero, levels, level_low, level_high, scale, eps)
            assert torch.isclose(test_output, torch.zeros_like(test_output))
    else:
        level_low = 0
        level_high = 255

        uniform_dist_input_low = Uniform(-100, 0)
        uniform_dist_input_range = Uniform(0, 100)
        for _ in range(number_of_samples):
            input_low = uniform_dist_input_low.sample().to(torch.device(device))
            input_range = uniform_dist_input_range.sample().to(torch.device(device))
            test_output = asymmetric_quantize(x_zero, levels, level_low, level_high, input_low, input_range, eps)
            assert torch.isclose(test_output, torch.zeros_like(test_output))


@pytest.mark.parametrize(
    "level_low, level_high, input_low, input_high, ref_zero_point, ref_scale",
    [
        [0, 255, 0.0548190139, 0.1554157883, 117, 0.0003944972],
        [0, 127, -0.0771205574, 0.0591347590, 72, 0.0010728766],
        [-128, 127, -0.1203568727, 0.0571926013, 45, 0.0006962725],
        [-64, 63, -0.0872629806, 0.2672979534, -33, 0.0027918185],
        [0, 255, 0.0, 1.0, 0, 0.0039215689],
        [0, 127, 0.0, 1.0, 0, 0.0078740157],
        [-128, 127, 0.0, 1.0, -128, 0.0039215689],
        [-64, 63, 0.0, 1.0, -64, 0.0078740157],
    ],
    ids=lambda v: str(v) if isinstance(v, int) else f"{v:.2f}",
)
def test_get_scale_zp_from_input_low_input_high(
    level_low, level_high, input_low, input_high, ref_zero_point, ref_scale
):
    scale, zero_point = get_scale_zp_from_input_low_input_high(
        torch.Tensor([level_low]), torch.Tensor([level_high]), torch.Tensor([input_low]), torch.Tensor([input_high])
    )
    assert zero_point == ref_zero_point, f"{zero_point} != {ref_zero_point}"
    assert np.isclose(scale, ref_scale), f"{scale:.10f} != {ref_scale}"
