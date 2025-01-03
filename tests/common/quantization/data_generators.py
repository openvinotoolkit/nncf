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
from typing import Dict, Tuple

import numpy as np


def get_quant_len_by_range(input_range: np.array, levels: int) -> np.array:
    """
    Returns quant length for N-bit quantization by input range.

    :param input_range: Input range of quantizer.
    :param levels: Number of quantization levels.

    :return np.array: Length of quant.
    """
    return input_range.astype(np.float64) / (levels - 1)


def get_quant_len(value_low: np.array, value_high: np.array, levels: int) -> np.array:
    """
    Returns quant length for N-bit quantization by min and max values.

    :param value_low: Low input value of quantizer.
    :param value_high: High input value of quantizer.
    :param levels: Number of quantization levels.

    :return np.array: Length of quant.
    """
    input_range = value_high.astype(np.float64) - value_low
    return get_quant_len_by_range(input_range, levels)


def get_quant_points(value_low: np.array, quant_len: np.array, levels: int) -> np.array:
    """
    Returns array of quant points for quantization parameters. Quant points are floating-point levels to which
    the corresponding input floating point values will be brought to after application of fake quantization.

    :param value_low: Low input value of quantizer.
    :param quant_len: Length of quant.
    :param levels: Number of quantization levels.

    :return np.array: Array of quant points.
    """
    return np.array([value_low.astype(np.float64) + i * quant_len.astype(np.float64) for i in range(levels)])


def get_mid_quant_points(value_low: np.array, quant_len: np.array, levels: int) -> np.array:
    """
    Returns an array of dots in the middle between quant points.

    :param value_low: Low input value of quantizer.
    :param quant_len: Length of quant.
    :param levels: Number of quantization levels.

    :return np.array: Array of quant points.
    """
    quant_points = get_quant_points(value_low, quant_len, levels)
    return (quant_points - quant_len / 2)[1:]


def generate_random_scale(min_scale: float = 0.1, max_scale: float = 1.0) -> float:
    """
    Generate random scale.

    :param min_scale: Minimum value for scale, defaults to 0.1
    :param max_scale: Maximum value for scale, defaults to 1.0

    :return float: Random scale value in the interval [min_scale, max_scale]
    """
    return min_scale + np.random.random() * (max_scale - min_scale)


def generate_random_scale_by_input_size(input_size: Tuple, is_per_channel: bool, is_weights: bool) -> np.array:
    """
    Generate random scales for each channels.

    :param input_size: Size of input tensor.
    :param is_per_channel: `True` for per-channel quantization, `False` for per-tensor.
    :param is_weights: Boolean that defines tensor type. True for Weights, False for Activations.

    :return np.array: Generated array of scale values.
    """
    if not is_per_channel:
        return np.array([generate_random_scale()])

    if is_per_channel:
        if is_weights:
            channel_count = input_size[0]
            scales_shape = [1 for _ in input_size]
            scales_shape[0] = channel_count
            scales = np.empty(scales_shape)
            for idx in range(0, channel_count):
                scales[idx] = generate_random_scale()
        else:
            channel_count = input_size[1]
            scales_shape = [1 for _ in input_size]
            scales_shape[1] = channel_count
            scales = np.empty(scales_shape)
            for idx in range(0, channel_count):
                scales[0, idx] = generate_random_scale()
    return scales


def generate_random_low_and_range(min_range: float = 0.1, max_range: float = 3.0) -> Tuple[float, float]:
    """
    Generate random input_low and input_range.

    :param min_range: Minimum value for range, defaults to 0.1
    :param max_range: Maximum value for range, defaults to 3.0

    :return Tuple[float, float]: input_low, input_range
    """
    input_low = np.random.random_sample() * max_range - max_range / 2
    input_range = min_range + np.random.random_sample() * max_range
    return input_low, input_range


def generate_random_low_and_range_by_input_size(
    input_size: Tuple, is_per_channel: bool, is_weights: bool
) -> Tuple[np.array, np.array]:
    """
    Generate random input_low and input_range for each channels.

    :param input_size: Size of input tensor.
    :param is_per_channel: `True` for per-channel quantization, `False` for per-tensor.
    :param is_weights: Boolean that defines tensor type. True for Weights, False for Activations.

    :return Tuple[np.array, np.array]: input_low, input_range
    """
    if not is_per_channel:
        input_low, input_range = generate_random_low_and_range()
        return np.array([input_low]), np.array([input_range])

    if is_per_channel:
        if is_weights:
            channel_count = input_size[0]
            scales_shape = [1 for _ in input_size]
            scales_shape[0] = channel_count
            input_low = np.empty(scales_shape)
            input_range = np.empty(scales_shape)
            for idx in range(0, channel_count):
                input_low[idx], input_range[idx] = generate_random_low_and_range()
        else:
            channel_count = input_size[1]
            scales_shape = [1 for _ in input_size]
            scales_shape[1] = channel_count
            input_low = np.empty(scales_shape)
            input_range = np.empty(scales_shape)
            for idx in range(0, channel_count):
                input_low[0, idx], input_range[0, idx] = generate_random_low_and_range()

    return input_low, input_range


def get_points_near_of_mid_points(input_data: np.array, mid_points: np.array, atol: float = 0.00005) -> np.array:
    """
    Get array of where 'True' means that point is in the middle between quant points.

    :param input_data: Input data.
    :param mid_points: An array of midpoints between quant points.
    :param atol: The absolute tolerance parameter for mid points, defaults to 0.00005.

    :return np.array: Array of flags to indicate points is in the middle between quant points.
    """
    num_elements = np.prod(input_data.shape)
    is_near_mid_point = np.zeros(num_elements).astype(bool)

    mid_point_ind = 0
    for ind in range(num_elements):
        value = input_data[ind]
        while mid_point_ind < len(mid_points) - 1 and mid_points[mid_point_ind] <= value - atol:
            mid_point_ind += 1

        if np.isclose(value, mid_points[mid_point_ind], atol=atol):
            is_near_mid_point[ind] = True
    return is_near_mid_point


def generate_lazy_sweep_data(shape: Tuple[int], min_val: float = -1.0, max_val: float = 1.0):
    """
    Generate tensor that contains sweep values from -1.0 to 1.0.

    :param shape: Shape of generate tensor.
    :param min_val: Min value of generated data.
    :param max_val: Max value of generated data.

    :return torch.Tensor: Generated tensor.
    """
    n = np.prod(list(shape))
    res = np.array(range(n)) / (n - 1) * (max_val - min_val) + min_val
    res[n // 2] = 0.0
    return res.reshape(shape)


def generate_sweep_for_one_channel(
    input_low, input_range, input_size, levels, rtol_for_mid_point
) -> Tuple[np.array, np.array, np.array]:
    """
    Generate sorted array that include:
        - values from `input_low - input_range * 0.5` to `input_low + input_range * 1.05`
          with fixed interval that depends from input_size;
        - quant points;
        - mid quant points;
        - values out of range;
        - zero.

    :param input_low: Array with low values.
    :param input_range: Array with range values.
    :param input_size: Size of input tensor.
    :param bits: Number of bits of quantization.
    :param rtol_for_mid_point: Relative tolerant value for points is in the middle between quant points.

    :return Tuple[np.array, np.array, np.array]: input_data, is_near_mid_point, quant_lens
    """
    input_low = input_low.astype(np.float64)
    input_range = input_range.astype(np.float64)

    min_val = input_low
    max_val = input_low + input_range
    quant_len = get_quant_len_by_range(input_range, levels)

    quant_points = get_quant_points(min_val, quant_len, levels)
    quant_mid_points = get_mid_quant_points(min_val, quant_len, levels)

    out_of_range_points = [
        min_val - quant_len,
        min_val - quant_len / 2,
        max_val + quant_len / 2,
        max_val + quant_len,
    ]

    result = np.concatenate([quant_points, quant_mid_points, out_of_range_points, [0.0]])

    num_elements = np.prod(input_size)
    assert len(result) < num_elements

    num = num_elements - len(result)
    ratio_out_of_range = 0.1
    g_range = input_range * (1.0 + 0.1)
    g_min = input_low - input_range * ratio_out_of_range / 2
    interval_points = np.array(np.arange(num) / (num - 1) * g_range + g_min)

    result = np.sort(np.concatenate([result, interval_points]))
    is_near_mid_point = get_points_near_of_mid_points(result, quant_mid_points, quant_len * rtol_for_mid_point)
    quant_lens = np.repeat(quant_len, num_elements)

    input_data = result.reshape(input_size)
    is_near_mid_point = is_near_mid_point.reshape(input_size)
    quant_lens = quant_lens.reshape(input_size)

    return input_data, is_near_mid_point, quant_lens


def generate_sweep_data(
    input_size: Tuple,
    input_low: np.array,
    input_range: np.array,
    levels: int,
    is_per_channel: bool,
    is_weights: bool,
    rtol_for_mid_point: float = 0.00005,
) -> np.array:
    """
    Generate sweep data by parameters.

    :param input_size: Size of input tensor.
    :param input_low: Array with low values.
    :param input_range: Array with range values.
    :param levels: Number of levels of quantization.
    :param is_per_channel: `True` for per-channel quantization, `False` for per-tensor.
    :param is_weights: Boolean that defines tensor type. True for Weights, False for Activations.
    :param rtol_for_mid_point: Relative tolerant value for points is in the middle between quant points,
        defaults to 0.00005.

    :return np.array: inputs, is_near_mid_point, quant_lens
    """
    if not is_per_channel:
        return generate_sweep_for_one_channel(input_low, input_range, input_size, levels, rtol_for_mid_point)

    inputs = None
    is_near_mid_point = None
    quant_lens = None
    if is_per_channel:
        if is_weights:
            channel_count = input_size[0]
            inputs = np.empty(input_size)
            is_near_mid_point = np.zeros(input_size).astype(bool)
            quant_lens = np.empty(input_size)
            for idx in range(0, channel_count):
                ch_input, ch_is_near_mid_point, ch_quant_lens = generate_sweep_for_one_channel(
                    input_low[idx], input_range[idx], input_size[1:], levels, rtol_for_mid_point
                )
                inputs[idx] = ch_input
                is_near_mid_point[idx] = ch_is_near_mid_point
                quant_lens[idx] = ch_quant_lens
        else:
            channel_count = input_size[1]
            inputs = np.empty(input_size)
            is_near_mid_point = np.zeros(input_size).astype(bool)
            quant_lens = np.empty(input_size)
            for idx in range(0, channel_count):
                ch_input, ch_is_near_mid_point, ch_quant_lens = generate_sweep_for_one_channel(
                    input_low[idx], input_range[idx], input_size[0:1] + input_size[2:], levels, rtol_for_mid_point
                )
                inputs[:, idx] = ch_input
                is_near_mid_point[:, idx] = ch_is_near_mid_point
                quant_lens[:, idx] = ch_quant_lens

    return inputs, is_near_mid_point, quant_lens


def check_outputs(arr_a: np.array, arr_b: np.array, is_near_mid_point: np.array, quant_lens: np.array, atol=0.000001):
    """
    Comparing values in arr_a and arr_b, with tolerant mismatch for points in the middle between quant points.

    :param arr_a: Data array.
    :param arr_b: Data array.
    :param is_near_mid_point: Array of that point is in the middle between quant points.
    :param quant_lens: Array of quant length that used as tolerance parameter for points in the middle
        between quant points.
    :param atol: The absolute tolerance parameter, defaults to 0.000001.

    :raises ValueError: If the arrays arr_a and arr_b do not match.
    """
    assert arr_a.shape == arr_b.shape
    assert arr_a.shape == is_near_mid_point.shape
    assert arr_a.shape == quant_lens.shape

    arr_a = arr_a.reshape(-1)
    arr_b = arr_b.reshape(-1)
    is_near_mid_point = is_near_mid_point.reshape(-1)
    quant_lens = quant_lens.reshape(-1)

    arr_diff = np.abs(arr_a - arr_b)
    arr_diff_points = arr_diff[np.invert(is_near_mid_point)]

    arr_diff_spec_points = arr_diff[is_near_mid_point]
    quant_lens_spec_points = quant_lens[is_near_mid_point]

    isclose_points = np.isclose(arr_diff_points, 0.0, atol=atol)
    max_error_for_spec_points = quant_lens_spec_points
    isclose_spec_points = np.isclose(
        np.maximum(arr_diff_spec_points, max_error_for_spec_points), max_error_for_spec_points, atol=atol
    )

    num_fail_points = sum(np.invert(isclose_points))
    num_fail_spec_points = sum(np.invert(isclose_spec_points))

    if num_fail_points or num_fail_spec_points:
        raise ValueError(
            f"Points: {num_fail_points} / {len(isclose_points)} | max_d={arr_diff_points.max():.8f} "
            f"Mid_points: {num_fail_spec_points} / {len(isclose_spec_points)} | max_d={arr_diff_spec_points.max():.8f}"
        )


def scatter_plot(
    data: Dict[str, np.array], x_column: str = None, vertical_lines: np.array = None, save_to_file: str = None
) -> None:
    """
    Function to render test data on scatter plot.
    If save_to_file is None, scatter plot will be rendered in interactive mode with enable zooming.

    Example of using:
    ```
    from tests.common.quantization.data_generators import scatter_plot

    scatter_plot(
        data={
            "input": test_input.detach().numpy(),
            "x_nncf":  x_nncf.detach().numpy(),
            "x_torch":  x_torch.detach().numpy(),
        },
        x_column="input",
        vertical_lines=mid_quant_points,
        save_to_file="plot.png"
    )
    ```

    :param data: Dict with data.
    :param x_column: Column name that will used as x-axis, defaults is 1st column.
    :param vertical_lines: Array to render vertical lines, defaults to None.
    :param save_to_file: Save plot to file as image, defaults to None.
    """
    import pandas as pd
    import plotly.express as px

    column_names = list(data.keys())
    for column in column_names:
        data[column] = data[column].reshape(-1)

    df = pd.DataFrame.from_dict(data)
    df["X"] = df[x_column if x_column else column_names[0]]

    fig = px.scatter(data_frame=df, x="X", y=column_names)
    fig.update_traces(marker={"size": 2})

    if vertical_lines is not None:
        for x_line in vertical_lines:
            fig.add_vline(x=x_line, line_width=1)

    if save_to_file:
        fig.write_image(save_to_file, scale=2)
    else:
        fig.show(config={"scrollZoom": True})
