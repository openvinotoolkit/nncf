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

from typing import TypeVar

from nncf.experimental.quantization.algorithms.range_estimator.algorithm import MinMaxRangeEstimator
from nncf.experimental.quantization.quantizer import Quantizer
from nncf.quantization.advanced_parameters import AdvancedBiasCorrectionParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.advanced_parameters import RangeEstimatorParameters
from nncf.quantization.algorithms.bias_correction.algorithm import BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FAST_BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.pipeline import Pipeline
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant

TModel = TypeVar("TModel")


def experimental_create_ptq_pipeline(
    quantizer: Quantizer,
    subset_size: int = 300,
    fast_bias_correction: bool | None = True,
    smooth_quant: bool = False,
    bias_correction_params: AdvancedBiasCorrectionParameters | None = None,
    smooth_quant_params: AdvancedSmoothQuantParameters | None = None,
    activations_range_estimator_params: RangeEstimatorParameters | None = None,
    weights_range_estimator_params: RangeEstimatorParameters | None = None,
    batchwise_statistics: bool = False,
) -> Pipeline:
    """
    Creates an experimental post-training quantization pipeline.

    The experimental post-training quantization pipeline includes the following steps:
        1) SmoothQuant
        2) MinMaxRangeInit
        3) FastBiasCorrection or BiasCorrection

    :param quantizer: Quantizer to use in MiMaxRangeInit algorithm.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param fast_bias_correction: Setting this option to `False` enables a different
        bias correction method which is more accurate, in general, and takes
        more time but requires less memory. None disables the bias correction algorithm.
    :param smooth_quant: Setting this option to `True` enables the SmoothQuant algorithm.
    :param bias_correction_params: Contains advanced parameters for fine-tuning bias correction algorithm.
    :param smooth_quant_params: Contains advanced alpha parameters for SmoothQuant algorithm.
    :param activations_range_estimator_params: Contains parameters for estimating the range
        of activations of the model.
    :param weights_range_estimator_params: Contains parameters for estimating the range
        of weights of the model.
    :param batchwise_statistics: Determines whether quantizer statistics should be calculated
        for each item of the batch or for the entire batch, default is False.
    :return: An experimental post-training quantization pipeline.
    """
    # Build the post-training quantization pipeline.
    pipeline_steps = []

    if smooth_quant_params is None:
        smooth_quant_params = AdvancedSmoothQuantParameters()

    if smooth_quant and (smooth_quant_params.convolution >= 0 or smooth_quant_params.matmul >= 0):
        alpha_map = {"convolution": smooth_quant_params.convolution, "matmul": smooth_quant_params.matmul}
        pipeline_steps.append([SmoothQuant(subset_size, False, alpha_map=alpha_map)])

    # Add the `MinMaxQuantization` algorithm as the third step of the pipeline.
    pipeline_steps.append(
        [
            MinMaxRangeEstimator(
                quantizer=quantizer,
                subset_size=subset_size,
                inplace_statistics=False,
                batchwise_statistics=batchwise_statistics,
                activations_range_estimator_params=activations_range_estimator_params,
                weights_range_estimator_params=weights_range_estimator_params,
            )
        ]
    )

    if fast_bias_correction is not None:
        # Add the `FastBiasCorrection` or `BiasCorrection` as additional algorithm
        # inside the third step of the pipeline. It is added after `MinMaxQuantization`
        # algorithm.
        if fast_bias_correction:
            threshold = FAST_BIAS_CORRECTION_THRESHOLD
            bias_correction_subset_size = subset_size
            bias_correction_cls = FastBiasCorrection
        else:
            threshold = BIAS_CORRECTION_THRESHOLD
            bias_correction_subset_size = max(int(subset_size * 0.2), 1)
            bias_correction_cls = BiasCorrection

        if bias_correction_params is None:
            bias_correction_params = AdvancedBiasCorrectionParameters()

        if bias_correction_params.threshold is not None:
            threshold = bias_correction_params.threshold

        pipeline_steps[-1].append(
            bias_correction_cls(
                bias_correction_subset_size,
                threshold,
                bias_correction_params.apply_for_all_nodes,
            )
        )

    return Pipeline(pipeline_steps)
