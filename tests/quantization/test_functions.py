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

import numpy as np
import pytest
import torch
from torch.autograd import Variable

from nncf.quantization.quantize_functions import asymmetric_quantize, symmetric_quantize, blockfp_quantize
from nncf.utils import sum_like
from tests.helpers import get_grads, check_equal
from ctypes import c_float, c_int32, cast, pointer, POINTER

EPS = 1e-6


class ReferenceQuantize:
    @staticmethod
    def forward(input_, input_low, input_range, levels):
        scale = (levels - 1) / input_range
        output = input_.clip(min=input_low, max=input_low + input_range)
        output -= input_low
        output *= scale
        output = output.round()
        output = output / scale
        output += input_low

        return output

    @staticmethod
    def backward(grad_output, input_, input_low, input_range, output, level_low, level_high, range_sign):
        mask_hi = (input_ > (input_low + input_range)).astype(input_.dtype)
        mask_lo = (input_ < input_low).astype(input_.dtype)

        mask_in = 1 - mask_hi - mask_lo
        err = (output - input_) * np.reciprocal(input_range * range_sign)
        grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
        grad_range = sum_like(grad_range, input_range)

        grad_input = grad_output * mask_in

        grad_low = grad_output * (mask_hi + mask_lo)
        grad_low = sum_like(grad_low, input_low)
        return [grad_input, grad_low, grad_range]

    @staticmethod
    def tune_range(input_low, input_range, levels):
        input_high = input_range + input_low
        input_low[input_low > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        scale = levels / (input_high - input_low)
        zp = np.round(-input_low * scale)

        new_input_low = np.where(zp < n, zp / (zp - n) * input_high, input_low)
        new_input_high = np.where(zp > 0., (zp - n) / zp * input_low, input_high)

        range_1 = input_high - new_input_low
        range_2 = new_input_high - input_low

        mask = (range_1 > range_2)
        inv_mask = abs(1 - mask)

        new_input_low = mask * new_input_low + inv_mask * input_low
        new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

        return new_input_low, new_input_range


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def idfn(val):
    if isinstance(val, list):
        return '[{}]'.format('-'.join([str(v) for v in val]))

    return None


@pytest.fixture
def _seed():
    np.random.seed(0)


def generate_input(input_size, dtype="float64"):
    return 1.0 * (2 * np.random.random_sample(input_size) - 1)


def get_test_data(data_list, is_cuda=False, is_backward=False, is_fp16=False):
    results = []
    for data in data_list:
        result = torch.from_numpy(data.copy())
        if is_cuda:
            result = result.cuda()
        if is_fp16:
            result = result.half()
        if is_backward:
            result = Variable(result, requires_grad=True)
        results.append(result)
    return results


def skip_if_half_on_cpu(is_fp16, use_cuda):
    if is_fp16 and not use_cuda:
        pytest.skip("As of PyTorch 1.5, the 'abs' operation is not supported on CPU for half and therefore"
                    "symmetric quantize fails. Remove this once this is fixed in PyTorch.")


def check_outputs_for_quantization_functions(test_val: torch.Tensor, ref_val: np.ndarray, is_fp16, rtol=1e-4):
    if is_fp16:
        # FP16 seems to be so inaccurate that ref and test quantization runs
        # will never be aligned quantum-wise - for quanta close to the
        # distribution's zero point even a difference in 1 quantum gives a
        # 100% relative error, and this is exactly what happens in FP16 for ~5% of the
        # input values uniformly sampled from the [-1.0; 1.0]. Therefore won't check for
        # tensor equality - the test passes for FP32 cases, and the kernel implementation
        # is exactly the same for FP16 calculations-wise.
        return
    check_equal(test_val, ref_val, rtol)


@pytest.mark.parametrize('input_size',
                         [[1, 48, 112, 112],
                          [1, 96, 28, 28],
                          [1, 288, 14, 14],
                          [16, 96, 112, 112],
                          [16, 192, 28, 28],
                          [16, 576, 14, 14]],
                         ids=idfn)
@pytest.mark.parametrize('bits', (8, 4), ids=('8bit', '4bit'))
@pytest.mark.parametrize("use_cuda", [False, True], ids=['cpu', 'cuda'])
@pytest.mark.parametrize('scale_mode', ["single_scale", "per_channel_scale"])
@pytest.mark.parametrize("is_weights", (True, False), ids=('weights', 'activation'))
@pytest.mark.parametrize("is_fp16", (True, False), ids=('fp16', 'fp32'))
class TestParametrized:
    @pytest.mark.parametrize("is_signed", (True, False), ids=('signed', 'unsigned'))
    class TestSymmetric:
        @staticmethod
        def generate_scale(input_, scale_mode, is_weights):
            assert scale_mode in ["single_scale", "per_channel_scale"]

            def calc_scale(input_):
                # Should generate a scale that is 1/2 of the input data span,
                # to test the out-of-bounds gradient calculation
                return (min(abs(input_.min()), abs(input_.max())) - input_.mean()) / 4

            if scale_mode == "single_scale":
                return np.array([calc_scale(input_)])

            if scale_mode == "per_channel_scale":
                if is_weights:
                    channel_count = input_.shape[0]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[0] = channel_count
                    scales = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[idx, ...]
                        scales[idx] = calc_scale(single_input_channel)
                else:
                    channel_count = input_.shape[1]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[1] = channel_count
                    scales = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[:, idx, ...]
                        scales[0, idx] = calc_scale(single_input_channel)
                return scales

        @staticmethod
        def get_range_level(is_signed, is_weights, bits):
            levels = 2 ** bits
            if is_signed:
                if is_weights:
                    levels -= 1
                level_high = 2 ** (bits - 1) - 1
                level_low = -(level_high + 1)
                if is_weights:
                    level_low += 1
            else:
                level_high = 2 ** bits - 1
                level_low = 0
            return level_low, level_high, levels

        def test_quantize_symmetric_forward(self, _seed, is_signed, is_weights, is_fp16, input_size, bits, use_cuda,
                                            scale_mode):
            skip_if_half_on_cpu(is_fp16, use_cuda)
            ref_input = generate_input(input_size)

            ref_scale = self.generate_scale(ref_input, scale_mode, is_weights)

            if is_fp16:
                ref_input = ref_input.astype(np.float16)
                ref_scale = ref_scale.astype(np.float16)

            test_input, test_scale = get_test_data([ref_input, ref_scale], use_cuda, is_fp16=is_fp16)
            level_low, level_high, levels = self.get_range_level(is_signed, is_weights, bits)

            ref_scale = abs(ref_scale) + EPS

            ref_input_low = ref_scale * (level_low / level_high)
            ref_input_range = ref_scale - ref_input_low

            ref_value = ReferenceQuantize.forward(ref_input, ref_input_low, ref_input_range, levels)

            test_value = symmetric_quantize(test_input, levels, level_low, level_high, test_scale, EPS)

            check_outputs_for_quantization_functions(test_value, ref_value, is_fp16, rtol=1e-2 if is_fp16 else 1e-3)

        def test_quantize_symmetric_backward(self, _seed, is_signed, is_weights, is_fp16, input_size, bits, use_cuda,
                                             scale_mode):
            skip_if_half_on_cpu(is_fp16, use_cuda)
            ref_input = generate_input(input_size)

            ref_scale = self.generate_scale(ref_input, scale_mode, is_weights)
            level_low, level_high, levels = self.get_range_level(is_signed, is_weights, bits)
            test_input, test_scale = get_test_data([ref_input, ref_scale], use_cuda, is_backward=True,
                                                   is_fp16=is_fp16)

            ref_scale = abs(ref_scale) + EPS
            if is_fp16:
                ref_input = ref_input.astype(np.float16)
                ref_scale = ref_scale.astype(np.float16)

            ref_input_low = ref_scale * (level_low / level_high)
            ref_input_range = ref_scale - ref_input_low

            ref_output = ReferenceQuantize.forward(ref_input, ref_input_low, ref_input_range, levels)

            mock_prev_output_grads = np.ones(input_size, dtype=np.float16 if is_fp16 else np.float)
            ref_grads = ReferenceQuantize.backward(mock_prev_output_grads, ref_input, ref_input_low,
                                                   ref_input_range, ref_output, level_low, level_high,
                                                   True)
            del ref_grads[1]
            test_value = symmetric_quantize(test_input, levels, level_low, level_high, test_scale, EPS)
            test_value.sum().backward()
            test_grads = get_grads([test_input, test_scale])

            check_outputs_for_quantization_functions(test_value, ref_output, is_fp16)
            check_outputs_for_quantization_functions(test_grads, ref_grads, is_fp16)

    class TestAsymmetric:
        @staticmethod
        def generate_range(input_, scale_mode, is_weights, is_fp16):
            assert scale_mode in ["single_scale", "per_channel_scale"]

            def calc_low_and_range(input_, is_fp16):
                # Should generate input_low and input_range that cover only the internal
                # 3/4 of the input data span to test the out-of-bounds gradient calculation
                span = input_.max() - input_.min()
                input_low = input_.min() + span / 8
                input_range = span * 3 / 4

                if is_fp16:
                    input_low = input_low.astype(np.float16)
                    input_range = input_range.astype(np.float16)

                return input_low, input_range

            if scale_mode == "single_scale":
                input_low, input_range = calc_low_and_range(input_, is_fp16)
                return np.array([input_low]), np.array([input_range])

            if scale_mode == "per_channel_scale":
                if is_weights:
                    channel_count = input_.shape[0]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[0] = channel_count
                    input_low = np.zeros(scales_shape)
                    input_range = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[idx, ...]
                        input_low[idx], input_range[idx] = calc_low_and_range(single_input_channel,
                                                                              is_fp16)
                else:
                    channel_count = input_.shape[1]
                    if channel_count == 1:
                        pytest.skip("Same case as for single scale mode")
                    scales_shape = [1 for _ in input_.shape]
                    scales_shape[1] = channel_count
                    input_low = np.zeros(scales_shape)
                    input_range = np.zeros(scales_shape)
                    for idx in range(0, channel_count):
                        single_input_channel = input_[:, idx, ...]
                        input_low[0, idx], input_range[0, idx] = calc_low_and_range(single_input_channel, is_fp16)

                return input_low, input_range

        @staticmethod
        def get_range_level(bits):
            levels = 2 ** bits
            level_low = 0
            level_high = levels - 1
            return level_low, level_high, levels

        def test_quantize_asymmetric_forward(self, _seed, input_size, bits, use_cuda, is_weights,
                                             is_fp16, scale_mode):
            skip_if_half_on_cpu(is_fp16, use_cuda)
            level_low, level_high, levels = self.get_range_level(bits)
            ref_input = generate_input(input_size)
            if is_fp16:
                ref_input = ref_input.astype(np.float16)

            ref_input_low, ref_input_range = self.generate_range(ref_input, scale_mode, is_weights,
                                                                 is_fp16)
            test_input, test_input_low, test_input_range = get_test_data(
                [ref_input, ref_input_low, ref_input_range], use_cuda, is_fp16=is_fp16)

            ref_input_range = abs(ref_input_range) + EPS
            ref_input_low, ref_input_range = ReferenceQuantize.tune_range(
                ref_input_low, ref_input_range, levels)
            ref_value = ReferenceQuantize.forward(
                ref_input, ref_input_low, ref_input_range, levels)
            test_value = asymmetric_quantize(test_input, levels, level_low, level_high, test_input_low,
                                             test_input_range, EPS)

            check_outputs_for_quantization_functions(test_value, ref_value, is_fp16)

        def test_quantize_asymmetric_backward(self, _seed, input_size, bits, use_cuda, is_weights,
                                              is_fp16, scale_mode):
            skip_if_half_on_cpu(is_fp16, use_cuda)
            level_low, level_high, levels = self.get_range_level(bits)
            ref_input = generate_input(input_size)
            if is_fp16:
                ref_input = ref_input.astype(np.float16)

            ref_input_low, ref_input_range = self.generate_range(ref_input, scale_mode, is_weights,
                                                                 is_fp16)
            test_input, test_input_low, test_input_range = get_test_data(
                [ref_input, ref_input_low, ref_input_range], use_cuda, is_backward=True, is_fp16=is_fp16)

            range_sign = np.sign(ref_input_range)
            ref_input_range = abs(ref_input_range) + EPS
            ref_input_low, ref_input_range = ReferenceQuantize.tune_range(
                ref_input_low, ref_input_range, levels)
            ref_output = ReferenceQuantize.forward(ref_input, ref_input_low, ref_input_range, levels)

            mock_prev_output_grads = np.ones(input_size, dtype=np.float16 if is_fp16 else np.float)
            ref_grads = ReferenceQuantize.backward(
                mock_prev_output_grads, ref_input, ref_input_low, ref_input_range, ref_output, level_low,
                level_high, range_sign)

            test_value = asymmetric_quantize(test_input, levels, level_low, level_high, test_input_low,
                                             test_input_range, eps=EPS)
            test_value.sum().backward()
            test_grads = get_grads([test_input, test_input_low, test_input_range])

            check_outputs_for_quantization_functions(test_grads, ref_grads, is_fp16)

def fold(src, fold_config):
    input_shape = src.shape

    num_in = input_shape[0]
    depth_in = input_shape[1]
    height_in = input_shape[2]
    width_in = input_shape[3]

    stride_x = fold_config['stride'][0]
    stride_y = fold_config['stride'][1]
    offset_x = fold_config['offset'][0]
    offset_y = fold_config['offset'][1]

    num_out = num_in
    depth_out = depth_in * stride_x * stride_y
    height_out = (height_in + offset_x + stride_x - 1) // stride_x
    width_out = (width_in + offset_y + stride_y - 1) // stride_y

    output_shape = (num_out, depth_out, height_out, width_out)

    output = np.zeros(output_shape)

    c_pos = 0
    for y in range(stride_y):
        raw_input_pos_y = y - offset_y
        input_pos_y = (raw_input_pos_y + stride_y) % stride_y
        output_pos_y = (input_pos_y - raw_input_pos_y) // stride_y
        slice_height = (height_in - input_pos_y + stride_y - 1)//stride_y
        for x in range(stride_x):
            raw_input_pos_x = x - offset_x
            input_pos_x = (raw_input_pos_x + stride_x) % stride_x
            output_pos_x = (input_pos_x - raw_input_pos_x) // stride_x
            slice_width = (width_in - input_pos_x + stride_x - 1)//stride_x
            output[0:num_out, c_pos:c_pos+depth_in, output_pos_y:output_pos_y+\
            slice_height, output_pos_x:output_pos_x+slice_width] = \
                src[0:num_in, 0:depth_in, input_pos_y:height_in:stride_y, input_pos_x:width_in:stride_x]
            c_pos = c_pos + depth_in



    return output

@pytest.mark.parametrize('input_size',
                         [[1, 3, 112, 112],
                          [1, 96, 14, 14],
                          [64, 3, 7, 7],
                          [2, 32, 56, 56]],
                         ids=idfn)
@pytest.mark.parametrize('bits', (4, 5, 12), ids=('int4bfp', 'int5bfp', 'fp16bfp'))
@pytest.mark.parametrize('block_size', (1, 8, 32), ids=('block1', 'block8', 'block32'))
@pytest.mark.parametrize("is_weights", (True, False), ids=('weights', 'activation'))
@pytest.mark.parametrize("fold_config", [None,
                                         {'offset':[0, 0], 'stride':[2, 2]},
                                         {'offset':[1, 1], 'stride':[2, 2]}], ids=('not_folded', 'S22_O00', 'S22_O11'))
class TestBlockfp:
    @staticmethod
    def float_as_int(src):
        bits = cast(pointer(c_float(src)), POINTER(c_int32)).contents
        value = bits.value
        return value

    @staticmethod
    def float_to_parts(src):
        value = TestBlockfp.float_as_int(src)
        sign = (value >> 31) & 1
        exponent = (value >> 23) & 0xff
        mantissa = (value) & 0x7fffff
        return (sign, exponent, mantissa)

    @staticmethod
    def parts_to_float(sign, exponent, mantissa):
        bits = (sign << 31) | (exponent << 23) | mantissa
        tmp = cast(pointer(c_int32(bits)), POINTER(c_float)).contents
        return tmp.value

    @staticmethod
    def check_block(ref_block, dut_block, exponent_bits, bits, block_size):
        assert ref_block.shape == dut_block.shape
        ref_signs = np.sign(ref_block)
        dut_signs = np.sign(dut_block)
        assert np.amin(ref_signs * dut_signs) >= 0 # No signs are opposite. Some can be zero

        ref_max = np.amax(np.abs(ref_block))
        dut_max = np.amax(np.abs(dut_block))
        (_, ref_exponent, _) = TestBlockfp.float_to_parts(ref_max)
        (_, dut_exponent_max, dut_mantissa_max) = TestBlockfp.float_to_parts(dut_max)


        # exponent matches
        # or dut has rounded up to next power of 2
        # or it's a subnormal rounded to zero
        assert(ref_exponent == dut_exponent_max or \
              (ref_exponent == dut_exponent_max-1 and dut_mantissa_max == 0) or \
              (dut_exponent_max == 0 and ref_exponent <= 128-(1<<exponent_bits-1)))
        mantissa_bits = bits - 2
        assert mantissa_bits >= 0
        mantissa_mask = (1<<(23-mantissa_bits))-1
        if block_size > 1:
            for dut_val in dut_block:
                if dut_val:
                    (_, dut_exponent, dut_mantissa) = TestBlockfp.float_to_parts(dut_val)
                    assert dut_exponent <= dut_exponent_max
                    exponent_delta = dut_exponent_max - dut_exponent
                    mantissa_mask = (1<<(23-mantissa_bits-exponent_delta))-1
                    assert(dut_mantissa & mantissa_mask) == 0

            if dut_exponent_max != 0:
                # calculate the Effective quantization value - one lsb of mantissa
                # and quantize rounding down. Check dut is same as ref or
                # one less if bfp value rounded up.
                lsb = TestBlockfp.parts_to_float(0, dut_exponent_max-mantissa_bits, 0)
                scaled_ref = np.abs(ref_block) // lsb
                scaled_dut = np.abs(dut_block) // lsb

                assert((scaled_ref >= scaled_dut).all and (scaled_ref <= scaled_dut +1).all)

    @staticmethod
    def check_bfp_outputs_for_quantization_functions(ref, dut, exponent_bits, bits, block_size):
        assert ref.shape == dut.shape

        for n in range(ref.shape[0]):
            for y in range(ref.shape[2]):
                for x in range(ref.shape[3]):
                    for c in range(0, ref.shape[1], block_size):
                        ref_block = ref[n, c:c+block_size, y, x]
                        dut_block = dut[n, c:c+block_size, y, x]
                        TestBlockfp.check_block(ref_block, dut_block, exponent_bits, bits, block_size)

    def test_quantize_blockfp(self, _seed, input_size, bits, block_size, is_weights, fold_config):
        exponent_bits = 5
        ref_input = np.float32(generate_input(input_size))

        if fold_config is not None and fold_config['stride'][0]*fold_config['stride'][1]*input_size[1] > block_size:
            pytest.skip("Blocksize smaller than folding dimensions")

        if fold_config is not None and is_weights and (fold_config['stride'][0] or fold_config['stride'][1]):
            pytest.skip("Weights never have non zero offset")

        [test_input_cuda] = get_test_data([ref_input], is_cuda=True, is_fp16=False)#
        [test_input_cpu] = get_test_data([ref_input], is_cuda=False, is_fp16=False)#

        dut_output_cuda = blockfp_quantize(test_input_cuda, exponent_bits, bits, block_size,
                                           fold_config, is_weights, name="").cpu().numpy()
        dut_output_cpu = blockfp_quantize(test_input_cpu, exponent_bits, bits, block_size,
                                          fold_config, is_weights, name="").cpu().numpy()
        assert (dut_output_cuda == dut_output_cpu).all

        if fold_config is not None:
            folded_input = fold(ref_input, fold_config)
            folded_output = fold(dut_output_cpu, fold_config)
            self.check_bfp_outputs_for_quantization_functions(folded_input, folded_output, exponent_bits,
                                                              bits, block_size)
        else:
            self.check_bfp_outputs_for_quantization_functions(ref_input, dut_output_cpu, exponent_bits,
                                                              bits, block_size)
