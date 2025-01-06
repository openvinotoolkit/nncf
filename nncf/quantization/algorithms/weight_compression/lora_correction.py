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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import nncf
from nncf.common.logging import nncf_logger
from nncf.common.utils.debug import DEBUG_LOG_DIR
from nncf.common.utils.debug import is_debug
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedLoraCorrectionParameters
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_nf4_quantization
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType


class DebugInterface:
    """
    Utility class to collect and dump debug information of the Lora Correction algorithm.
    """

    def __init__(self):
        self._noise_per_layer = {}

    def add_noises(self, layer_name: str, value: float):
        self._noise_per_layer[layer_name] = value

    def dump_data(self):
        if not self._noise_per_layer:
            return
        dump_dir = Path(DEBUG_LOG_DIR) / "lora"
        dump_dir.mkdir(parents=True, exist_ok=True)

        layer_dir = dump_dir / "per_layer"
        layer_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._noise_per_layer)
        losses_path = dump_dir / "noises.csv"
        nncf_logger.debug(f"Quantization noise through the correction process is saved to: {losses_path}")
        df.to_csv(losses_path)

        for name in df.columns:
            plt.plot(df[name])
            filename = name.replace("/", "_") + ".jpg"
            plt.savefig(layer_dir / filename)
            plt.clf()

        delta = df.iloc[0] - df.iloc[-1]
        nncf_logger.debug(f"Is quantization noise reduced for all layers: {all(delta > 0)}")

        _, ax = plt.subplots(1)
        ax.plot(delta)
        ax.set_xticklabels([])
        delta_path = dump_dir / "qnoise_change.jpg"
        nncf_logger.debug(f"Saving change in quantization noise for each layer to: {delta_path}")
        plt.savefig(delta_path)
        plt.clf()


class LoraCorrectionAlgorithm:
    """
    Contains implementation of LoRA Correction algorithm.

    The method reduces quantization noise after weight compression using low rank adapters.
    """

    def __init__(
        self, statistics: Dict[str, WCTensorStatistic], lora_correction_params: AdvancedLoraCorrectionParameters
    ):
        """
        :param statistics: Input activation statistics for each node.
        :param lora_correction_params: parameters to configure the algorithm.
        """
        self._statistics = statistics
        self._lora_correction_params = lora_correction_params
        self._debug_interface = DebugInterface() if is_debug() else None

    def __del__(self):
        if self._debug_interface is not None:
            self._debug_interface.dump_data()

    @property
    def use_int8_adapters(self) -> bool:
        return self._lora_correction_params.use_int8_adapters

    def is_applicable(self, wc_params: WeightCompressionParameters):
        return wc_params.compression_config.num_bits == 4

    def calculate_adapters(
        self, weight: Tensor, compressed_weight: Tensor, wc_params: WeightCompressionParameters
    ) -> Tuple[Tensor, Tensor, List[float]]:
        """
        Calculates low rank matrices for a given original and compressed weights.

        :param weight: original floating-point weight matrix.
        :param compressed_weight: compressed weight matrix.
        :param wc_params: parameters of weight compression.
        :return: two low rank matrices in the order of execution of corresponding linear layers.
        """
        layer_name = wc_params.node_with_weight.node_name
        layer_statistics = self._statistics[layer_name]
        is_debug = self._debug_interface is not None
        lora_A, lora_B, mean_noises = self.calculate_low_rank_matrices(
            weight,
            compressed_weight,
            wc_params.compression_config,
            wc_params.reduction_axes,
            self._lora_correction_params,
            layer_statistics,
            is_debug,
        )
        if is_debug:
            self._debug_interface.add_noises(layer_name, mean_noises)
        return lora_A, lora_B

    @staticmethod
    def calculate_low_rank_matrices(
        weight: Tensor,
        compressed_weight: Tensor,
        compression_config: WeightCompressionConfig,
        reduction_axes: Tuple[int, ...],
        lora_correction_params: AdvancedLoraCorrectionParameters,
        layer_statistics: WCTensorStatistic,
        is_debug: Optional[bool] = False,
    ):
        """
        Calculates low rank matrices for a given original and compressed weights.
        The low rank matrices obtained by applying singular value decomposition (SVD) with lower rank for the
        difference between original weight and fake-quantized ones.
        Then, an iterative algorithm refines them. It solves a system of linear equations alternately fixing
        one matrix, then another.

        :param weight: original floating-point weight matrix.
        :param compressed_weight: compressed weight matrix.
        :param compression_config: configuration of weight compression for the weight node.
        :param reduction_axes: axes along which different statistics reduced.
        :param lora_correction_params: parameters to configure the algorithm.
        :param layer_statistics: an object containing statistics for the layer.
        :param is_debug: whether to collect debug information, defaults to False.
        :return: two low rank matrices in the order of execution of corresponding linear layers and list of mean noises.
            Noises are collected from each step of the algorithm if debug was enabled.
            First low rank matrix has shape=[R, H], the second - [O, R], where R - rank, H/O - hidden/output dimension.
        """
        rank, num_iterations, apply_regularization, subset_size = (
            lora_correction_params.adapter_rank,
            lora_correction_params.num_iterations,
            lora_correction_params.apply_regularization,
            lora_correction_params.subset_size,
        )
        mode = compression_config.mode
        assert len(reduction_axes) == 1, "Assumed a single reduction axis"
        reduction_axis = reduction_axes[0] if compression_config.group_size != -1 else -1
        if mode in (CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM):
            fq_weights = do_int_dequantization(
                compressed_weight.tensor,
                compressed_weight.scale,
                compressed_weight.zero_point,
                reduction_axis,
            )
        elif mode == CompressWeightsMode.NF4:
            indexes = do_nf4_quantization(compressed_weight.tensor, compressed_weight.scale, is_normalized_weight=True)
            fq_weights = do_nf4_dequantization(indexes, compressed_weight.scale, reduction_axis)
        else:
            raise nncf.InternalError(
                f"{mode.value} mode is invalid for Lora Correction algorithm. Supported modes: INT4_SYM, INT4_ASYM, NF4"
            )
        # fq_w + residual = w   =>  residual = w - fq_w
        svd_residual = fns.astype(weight - fq_weights, TensorDataType.float32)

        # O stands for output dimension, H - input dimension or hidden size, SS - samples size, R - rank.
        # reduction axes is all axes except output dimension in linear/conv layers.
        if reduction_axes[0] == 1:
            svd_residual = fns.transpose(svd_residual)
        residual = svd_residual.clone()  # [H, O]

        s, X = process_stats(layer_statistics, subset_size)  # [H], [H, SS]
        X = fns.transpose(X)  # [SS, H]
        if compression_config.group_size > 0:
            # Multiply residual of weights by maximum channel magnitude of activations normalized per quantization
            # group. As a consequence, weights corresponding to a "noisy" activations has a higher error to correct.
            # Empirically, it leads to a better accuracy.
            gs = compression_config.group_size
            n_gs = s.shape[0] // gs
            for i in range(n_gs):
                offset = i * gs
                denum = fns.sum(s[offset : offset + gs])
                s[offset : offset + gs] = s[offset : offset + gs] / denum
                denum = fns.max(s[offset : offset + gs])
                s[offset : offset + gs] = s[offset : offset + gs] / denum
            s = fns.expand_dims(s, 1)  # [H, 1]
            svd_residual = svd_residual * s  # [H, O]

        # Low-rank approximation.
        U_full, S_full, V_full = fns.linalg.svd(svd_residual, full_matrices=False)
        U = U_full[:, :rank]  # [H, R]
        S = fns.diag(S_full[:rank])  # [R, R]
        V = V_full[:rank, :]  # [R, O]
        V = S @ V  # [R, O]

        # An iterative algorithm for correction (refinement) of the low-rank adapters.
        mean_noises = []
        noise = X @ residual  # [SS, H] * [H, O] = [SS, O]
        for i in range(num_iterations):
            # Part 1: U is fixed, find V.
            XU = X @ U  # [SS, R]
            if not apply_regularization:
                # X @ U @ V = noise      ---> a @ x = b
                new_V = fns.linalg.lstsq(XU, noise, driver="gelsy")
            else:
                # 1) U @ V = res         <--- regularization
                # 2) X @ U @ V = noise
                # |U X @ U| @ V = |res noise|
                UXU = fns.concatenate([U, XU], axis=0)  # [H + SS, R]
                noiseR = fns.concatenate([residual, noise], axis=0)  # [H + SS, O]
                new_V = fns.linalg.lstsq(UXU, noiseR, driver="gelsy")
            if is_debug:
                if i == 0:
                    init_noise = noise
                    mean_noise_before_svd = fns.mean(fns.abs(init_noise)).item()
                    mean_noise_after_svd = fns.mean(fns.abs(init_noise - XU @ V)).item()
                    mean_noises.extend([mean_noise_before_svd, mean_noise_after_svd])
                mean_noise_after_correct = fns.mean(fns.abs(init_noise - XU @ new_V)).item()
                mean_noises.append(mean_noise_after_correct)
                nncf_logger.debug(
                    f"{i} Correction U: {mean_noise_before_svd}, {mean_noise_after_svd}, {mean_noise_after_correct}"
                )
            V = new_V

            # Part 2: V is fixed, find U.
            VI = fns.linalg.pinv(V)  # [O, R]
            noiseVI = noise @ VI  # [R, SS]
            if not apply_regularization:
                # VI = V^-1
                # 1) X @ U @ V = noise
                # 1) X @ U = noise @ VI     ---> a @ x = b
                U = fns.linalg.lstsq(X, noiseVI, driver="gelsy")
            else:
                # VI = V^-1, E - identity matrix
                # 1) U @ V = res           <--- regularization
                # 1) E @ U = res @ VI
                # 2) X @ U @ V = noise
                # 2) X @ U = noise @ VI
                # |E X| = | (UI @ res) (UI @ noise) |
                E = fns.eye(U.shape[0], backend=U.backend, dtype=U.dtype)  # [H, H]
                EX = fns.concatenate([E, X], axis=0)  # [H + SS, H]
                noiseR = fns.concatenate([residual @ VI, noiseVI], axis=0)  # [H + SS, R]
                U = fns.linalg.lstsq(EX, noiseR, driver="gelsy")
            if is_debug:
                mean_noise_after_correct = fns.mean(fns.abs(init_noise - X @ U @ V)).item()
                mean_noises.append(mean_noise_after_correct)
                nncf_logger.debug(
                    f"{i} Correction V: {mean_noise_before_svd}, {mean_noise_after_svd}, {mean_noise_after_correct}"
                )
        return fns.transpose(U), fns.transpose(V), mean_noises
