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

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, TypeVar

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import _calculate_normalized_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_float_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


class CodebookEstimation(Algorithm):
    """
    Codebook Estimation Algorithm for Neural Network Weight Compression.

    This algorithm performs codebook estimation for weight compression using modified weighted K-means clustering.
    It minimizes the difference between floating-point MatMul operations and MatMul operations with
    compressed weights by computing optimal codebooks and indexes for weight compression.

    The algorithm works in two modes:
    1. Per-weight mode: Estimates a separate codebook for each weight tensor.
    2. Per-group mode: Groups weights by name patterns (e.g., down_proj, up_proj) and estimates a
       shared codebook across all weights in the group.

    Algorithm Overview:
    -------------------
    1. Weight Normalization: Normalize weights using calculated quantization scales.
    2. K-Means Clustering: Apply weighted K-means clustering on normalized weights, where importance
       is derived from input activation statistics.
    3. Codebook Variants: Generate multiple codebook variants including:
       - K-means optimized centroids
       - Default uniform codebook
       - Linear range codebook
    4. Optimal Selection: Evaluate each variant by comparing MatMul outputs with activation statistics
       and select the codebook that minimizes reconstruction error.

    Mathematical Foundation:
    ------------------------
    The algorithm minimizes: ||W @ X - Q(W) @ X||, where:
    - W: Original floating-point weight matrix
    - Q(W): Quantized/compressed weight matrix using codebook
    - X: Input activation statistics
    - ||.||: L1 norm of the difference

    Key Features:
    -------------
    - Supports grouped quantization for large weight matrices
    - Uses activation-weighted importance during clustering
    - Optimizes for specific floating-point formats (e.g., FP8 E4M3)
    - Can process weights with group_size != -1 for per-weight quantization
    """

    def __init__(
        self, value_type: TensorDataType = TensorDataType.f8e4m3, across_blocks: bool = True, num_elements: int = 16
    ):
        """
        Initializes the Codebook Estimation algorithm.

        :param value_type: Target data type for codebook values. Default is TensorDataType.f8e4m3
            (8-bit floating point with 4 exponent bits and 3 mantissa bits).
        :param across_blocks: If True, estimates codebooks per group of weights sharing the same name pattern.
            If False, estimates a separate codebook for each individual weight tensor.
        :param num_elements: Number of centroids in the codebook. For 4-bit quantization, this is
            typically 16 (2^4). Each weight value will be mapped to one of these centroids.
        """
        super().__init__()

        self._value_type = value_type
        self._across_blocks = across_blocks
        self._num_elements = num_elements

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backend-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        else:
            msg = (
                "Cannot return backend-specific Codebook Estimation entity because"
                f" {model_backend.value} is not supported!"
            )
            raise nncf.UnsupportedBackendError(msg)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        statistics: dict[str, WCTensorStatistic],
        backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> dict[str, CompressedWeight]:
        """
        Estimates optimal codebook for weight compression.

        Minimizes difference between floating point MatMul and MatMul with compressed weights.
        The algorithm computes codebook and indexes for MatMul compression.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param backend_entity: Weight compression algorithm backend.
        :return: A dictionary that maps weight names to CompressedWeight with codebook,
            codebook indexes and scale.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)

        if self._across_blocks:
            return self.apply_across_blocks(model, graph, all_weight_params, statistics, backend_entity)

        res = dict()

        for wp in track(all_weight_params, description="Applying Codebook Estimation"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if config.num_bits != 4:
                res[weight_name] = CompressedWeight()
                continue

            stats = statistics[node_name]

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

            codebook = self.calculate_codebook(stats, weight, wp.reduction_axes, config, wp)
            res[weight_name] = CompressedWeight(None, None, None, codebook)

        return res

    def apply_across_blocks(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        statistics: dict[str, WCTensorStatistic],
        backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> dict[str, CompressedWeight]:
        """
        Estimates optimal codebook for a group of weights grouped by name (e.g., down_proj, up_proj).

        Minimizes difference between floating point MatMul and MatMul with compressed weights.
        The algorithm computes codebook and indexes for MatMul compression.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param backend_entity: Weight compression algorithm backend.
        :return: A dictionary that maps weight names to CompressedWeight with codebook,
            codebook indexes and scale.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)
        res = dict()

        for wp in track(all_weight_params, description="Applying Codebook Estimation across blocks"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if weight_name in res:
                continue

            if config.num_bits != 4:
                res[weight_name] = CompressedWeight()
                continue

            weight = self.get_weight(model, graph, wp)
            if weight is None:
                continue

            weights = [weight]
            stats = [statistics[node_name]]
            group_weights_params = [wp]

            clear_weight_name = "".join(filter(lambda x: x.isalpha(), weight_name))

            for other_wp in all_weight_params:
                if other_wp.weight_name == weight_name:
                    continue
                other_weight_name = "".join(filter(lambda x: x.isalpha(), other_wp.weight_name))
                other_node_name = other_wp.node_with_weight.node_name

                if clear_weight_name == other_weight_name:
                    other_weight = self.get_weight(model, graph, other_wp)
                    if other_weight is not None:
                        weights.append(other_weight)
                        stats.append(statistics[other_node_name])
                        group_weights_params.append(other_wp)

            codebook = self.calculate_codebook_for_group(stats, weights, wp.reduction_axes, config, wp)

            for gwp in group_weights_params:
                res[gwp.weight_name] = CompressedWeight(None, None, None, codebook)
                gwp.compression_config.codebook_values = codebook

        return res

    def get_weight(self, model: TModel, graph: NNCFGraph, wp: WeightCompressionParameters) -> Tensor:
        """
        Retrieves weight tensor from the model for a given weight parameter.

        :param model: Model containing the weights.
        :param graph: Model graph.
        :param wp: Weight compression parameters.
        :return: Weight tensor or None if not supported.
        """
        weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
        if len(weight_data) != 1:  # not supported by the algorithm
            return None
        _, weight_port_id = weight_data[0]

        return self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

    def calculate_codebook(
        self,
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        wp: WeightCompressionParameters,
    ) -> Tensor:
        """
        Calculate optimal codebook for a single weight tensor.

        This method implements the core codebook estimation algorithm for an individual weight tensor.
        It performs the following steps:

        1. Weight preprocessing: Transpose if needed and reshape for grouped quantization.
        2. Importance calculation: Compute importance weights from activation statistics.
        3. Weight normalization: Normalize weights using quantization scales.
        4. K-means clustering: Apply weighted K-means to find initial centroids.
        5. Variant generation: Create multiple codebook candidates.
        6. Optimal selection: Select codebook that minimizes MatMul reconstruction error.

        :param statistics: Input activation statistics for the layer containing statistical
            information used to compute importance weights.
        :param weight: The weight tensor to compress, will be converted to float32.
        :param reduction_axes: Tuple of axes along which to perform quantization reduction.
        :param config: Weight compression configuration containing parameters like group_size,
            num_bits, and quantization mode.
        :param wp: Weight compression parameters containing node and reduction information.
        :return: Optimal codebook tensor in the target data type with shape (num_elements,).
            This codebook contains the centroid values that minimize reconstruction error.
        """
        reduction_axis = reduction_axes[0]
        weight = deepcopy(weight.astype(TensorDataType.float32))

        s, X = process_stats(statistics, -1)

        if reduction_axis == 0:
            weight = fns.transpose(weight)
            reduction_axis = 1

        orig_shape = weight.shape

        if config.group_size != -1:
            weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)
            s = fns.unsqueeze(s, -2)
            s, _ = reshape_weight_for_grouped_quantization(s, reduction_axis, config.group_size)

        importance = fns.ones_like(weight)
        importance = importance * s

        scale = calculate_float_quantization_params(weight, reduction_axes, config, signed=True)
        norm_weight = _calculate_normalized_weight(weight, scale)

        codebook, indexes, variants = weights_clusterization_k_means(
            norm_weight, importance, n_centroids=self._num_elements
        )

        indexes = indexes.reshape(weight.shape)

        best_codebook = codebook.as_openvino_tensor().astype(self._value_type)

        diff = float("inf")

        if self._num_elements == config.get_numpy_codebook().size:
            variants[0] = fns.tensor(
                config.get_numpy_codebook().data, backend=weight.backend, dtype=TensorDataType.float16
            )
        variants[1] = fns.tensor(
            list(range(-self._num_elements // 2, self._num_elements - self._num_elements // 2)),
            backend=weight.backend,
            dtype=TensorDataType.float16,
        )

        weight = fns.reshape(weight, orig_shape)

        fp_outs = fns.matmul(weight, X)
        for var in variants:
            var = var.as_openvino_tensor().astype(self._value_type)
            config.codebook_values = Tensor(var)
            qw = float_quantize_dequantize_weight(weight, config, wp.reduction_axes)
            q_outs = fns.matmul(fns.reshape(qw, orig_shape), X)

            cur_diff = fns.mean(fns.abs(fp_outs - q_outs)).item()
            if cur_diff < diff:
                diff = cur_diff
                best_codebook = var

        return Tensor(best_codebook)

    def calculate_codebook_for_group(
        self,
        statistics: list[WCTensorStatistic],
        weights: list[Tensor],
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        wp: WeightCompressionParameters,
    ) -> Tensor:
        """
        Calculate optimal codebook for a group of weight tensors sharing the same name pattern.

        This method extends single-weight codebook estimation to handle multiple related weights
        (e.g., all down_proj or up_proj layers in a transformer model). It optimizes a shared
        codebook across all weights in the group by:

        1. Processing each weight: Normalize and compute importance for each weight tensor.
        2. Concatenation: Combine all normalized weights and importance matrices.
        3. Global clustering: Apply weighted K-means on the combined data.
        4. Joint optimization: Select codebook minimizing total reconstruction error across all weights.

        The algorithm weights each layer's contribution to the total error by the mean absolute
        value of its activation statistics, ensuring that more important layers have greater
        influence on codebook selection.

        :param statistics: List of activation statistics, one per weight tensor in the group.
        :param weights: List of weight tensors to compress with a shared codebook.
        :param reduction_axes: Tuple of axes along which to perform quantization reduction.
        :param config: Weight compression configuration shared across all weights.
        :param wp: Weight compression parameters for accessing reduction information.
        :return: Optimal shared codebook tensor in the target data type with shape (num_elements,).
            This codebook is used to compress all weights in the group.
        """
        reduction_axis = reduction_axes[0]

        norm_weight = []
        importances = []
        Xs = []
        fp_outs = []

        for stat, weight in zip(statistics, weights):
            weight = deepcopy(weight.astype(TensorDataType.float32))
            s, X = process_stats(stat, -1)
            Xs.append(X)

            if reduction_axis == 0:
                weight = fns.transpose(weight)
                reduction_axis = 1

            if config.group_size != -1:
                weight, reduction_axes = reshape_weight_for_grouped_quantization(
                    weight, reduction_axes, config.group_size
                )

            fp_outs.append(fns.matmul(weight, X))

            importance = fns.ones_like(weight)
            importance = importance * s
            importances.append(importance)

            scale = calculate_float_quantization_params(weight, reduction_axes, config, signed=False)
            norm_weight.append(_calculate_normalized_weight(weight, scale))

        norm_weight = fns.concatenate(norm_weight, axis=0)
        importance = fns.concatenate(importances, axis=0)

        codebook, _, variants = weights_clusterization_k_means(
            norm_weight, importance, n_centroids=self._num_elements, intervals=100000
        )

        best_codebook = codebook.as_openvino_tensor().astype(self._value_type)

        diff = float("inf")

        if self._num_elements == config.get_numpy_codebook().size:
            variants[0] = fns.tensor(
                config.get_numpy_codebook().data, backend=weight.backend, dtype=TensorDataType.float16
            )
        variants[1] = fns.tensor(
            list(range(-self._num_elements // 2, self._num_elements - self._num_elements // 2)),
            backend=weight.backend,
            dtype=TensorDataType.float16,
        )

        coeffs = [fns.mean(fns.abs(X)).item() for X in Xs]

        for var in variants:
            var = var.as_openvino_tensor().astype(self._value_type)
            config.codebook_values = Tensor(var)

            cur_diff = 0.0
            for weight, X, fp_out, c in zip(weights, Xs, fp_outs, coeffs):
                qw = float_quantize_dequantize_weight(weight, config, wp.reduction_axes)
                q_out = fns.matmul(qw, X)
                cur_diff += c * fns.mean(fns.abs(fp_out - q_out)).item()
            if cur_diff < diff:
                diff = cur_diff
                best_codebook = var

        return Tensor(best_codebook)

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return StatisticPointsContainer()


def round_to_left(quantiles, values) -> Tensor:
    """
    Assign values to quantile bins using binary search.

    This utility function assigns each value to the nearest left quantile bin using
    efficient binary search. It's used for assigning data points to histogram bins
    and for assigning values to their nearest centroids.

    :param quantiles: Sorted tensor of bin boundaries or centroid positions.
    :param values: Tensor of values to assign to bins.
    :return: Tensor of integer indices indicating which bin each value belongs to.
        Index i means the value falls between quantiles[i] and quantiles[i+1].
    """
    center_of_quantiles = 0.5 * (quantiles[1:] + quantiles[:-1])
    return fns.searchsorted(center_of_quantiles, values, side="left", sorter=None)


@dataclass
class KMeansAlgoData:
    """
    Data container for K-means algorithm histograms and statistics.

    This dataclass stores preprocessed data used during the K-means clustering process,
    including histogram representations that enable efficient centroid updates.

    :param centroids: Tensor of histogram bin centers, representing discrete value ranges
        in the input data distribution. Shape: (num_bins,).
    :param weighted_centroids: Tensor of weighted sums for each bin, computed as
        Σ(data_values * importance) for all values in the bin.
        Used to calculate new centroid positions. Shape: (num_bins,).
    :param weighted_importance: Optional tensor of total importance weights for each bin,
        computed as Σ(importance) for all values in the bin.
        Used as denominator in centroid updates. Shape: (num_bins,).
    """

    centroids: Tensor
    weighted_centroids: Tensor
    weighted_importance: Tensor | None = None


class KMeansWeighted:
    """
    Weighted K-means clustering algorithm with fixed centroids support.

    :param n_clusters: Number of clusters (codebook size), typically 16 for 4-bit quantization.
    :param max_iter: Maximum number of iterations for centroid refinement, default 300.
    :param variants: List of centroid configurations saved during optimization, used for
                 selecting the best codebook based on actual reconstruction error.
    :param centroids: Final centroid positions after convergence. Shape: (n_clusters,)
    """

    def __init__(self, n_clusters=8, max_iter=300):
        """
        :param n_clusters: Number of clusters (centroids) to find. For codebook estimation,
            this equals the number of representable values (e.g., 16 for 4-bit).
        :param max_iter: Maximum number of iterations for centroid updates. The algorithm
            may converge earlier if centroids stabilize (change < 0.00001).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.variants = []
        self.centroids = None

    @staticmethod
    def get_init(values, frequencies, n_clusters) -> Tensor:
        """
        Initialize centroids using quantile-based placement.

        This method places initial centroids at evenly-spaced quantiles of the weighted
        cumulative distribution. This approach ensures centroids start at representative
        positions across the data distribution.

        :param values: Sorted array of unique or binned data values.
        :param frequencies: Importance weights corresponding to each value.
        :param n_clusters: Number of centroids to initialize.
        :return: Tensor of initial centroid positions with shape (n_clusters,).
            First and last centroids are placed at min and max values.
            Interior centroids are placed between quantile-based positions.
        """
        step = 1.0 / (n_clusters - 1)
        denum = fns.sum(frequencies)
        quants = [i * step for i in range(n_clusters)]
        n_frequencies = frequencies / denum
        n_frequencies = fns.cumsum(n_frequencies, axis=0)

        res = fns.zeros((n_clusters,), backend=values.backend, dtype=values.dtype)
        for i in range(n_clusters):
            if i == 0:
                res[i] = values[0]
            elif i == n_clusters - 1:
                res[i] = values[-1]
            else:
                prev_val = values[fns.nonzero(n_frequencies <= quants[i])[0][-1].item()].item()
                next_val = values[fns.nonzero(n_frequencies <= quants[i + 1])[0][-1].item()].item()
                res[i] = (prev_val + next_val) / 2

        # avoid close centroids
        th = 0.05
        for i in range(1, n_clusters - 1):
            if (res[i] - res[i + 1]).abs() / max(res[i].abs(), res[i + 1].abs()) < th:
                res[i] = (res[i - 1] + res[i + 1]) / 2

        return res

    @staticmethod
    def create_histogramm_sorted(data_, importance, intervals=700) -> KMeansAlgoData:
        """
        Create a weighted histogram from sorted data for efficient K-means updates.

        This method preprocesses data into a histogram representation that dramatically
        reduces computational cost during iterative K-means updates. Instead of processing
        millions of weight values, the algorithm processes hundreds of histogram bins.

        Algorithm:
        1. Sort data and corresponding importance weights.
        2. Divide the data range into equal-width bins (intervals).
        3. For each bin, compute:
           - Centroid: middle of the bin range.
           - Weighted data: Σ(data * importance) for all values in bin.
           - Weighted importance: Σ(importance) for all values in bin.

        :param data_: Input data tensor to be histogrammed (will be sorted internally).
        :param importance: Importance weights for each data point.
        :param intervals: Number of histogram bins to create, default 700.
            Higher values provide more precision but slower updates.
        :return: KMeansAlgoData object containing centroids (bin center positions),
            weighted_centroids (weighted sum of data in each bin), and
            weighted_importance (total importance weight in each bin).
        """
        centers = []
        ranges = []

        step = data_.max().item() - data_.min().item()
        step /= intervals

        sorted_idx = fns.argsort(data_)
        data = data_[sorted_idx]
        importance = importance[sorted_idx]

        data_range = (data.min().item(), data.max().item())
        prev = data_range[0]

        while prev < data_range[1]:
            centers.append(prev + step / 2)
            prev += step

            if len(centers) > 1:
                ranges.append(0.5 * (centers[-2] + centers[-1]))
            ranges.append(centers[-1])

        centers = fns.tensor(centers, backend=data_.backend, dtype=data_.dtype)
        ranges = fns.tensor(ranges, backend=data_.backend, dtype=data_.dtype)

        ranges_idxs = round_to_left(data, ranges)

        res_centers = []
        weighted_data = []
        weighted_importance = []

        for i in range(centers.size):
            if i == 0:
                data_range, importance_range = data[: ranges_idxs[1].item()], importance[: ranges_idxs[1].item()]
            elif i == centers.size - 1:
                data_range, importance_range = data[ranges_idxs[-2].item() :], importance[ranges_idxs[-2].item() :]
            else:
                idx = 2 * i
                data_range, importance_range = (
                    data[ranges_idxs[idx - 1].item() : ranges_idxs[idx + 1].item()],
                    importance[ranges_idxs[idx - 1].item() : ranges_idxs[idx + 1].item()],
                )

            if data_range.size == 0:
                continue
            res_centers.append(centers[i].item())
            weighted_data.append(fns.sum(fns.multiply(data_range, importance_range)).item())
            weighted_importance.append(fns.sum(importance_range).item())

        res = KMeansAlgoData(
            fns.tensor(res_centers, backend=data_.backend, dtype=data_.dtype),
            fns.tensor(weighted_data, backend=data_.backend, dtype=data_.dtype),
            fns.tensor(weighted_importance, backend=data_.backend, dtype=data_.dtype),
        )
        return res

    def fit(self, X_train, importance, init, fixed=None, intervals=700) -> None:
        """
        Fit the weighted K-means model to training data with optional fixed centroids.

        This method performs iterative refinement of centroids while respecting constraints
        on certain centroid positions (e.g., keeping min, zero, and max fixed).

        Algorithm Flow:
        1. Preprocessing: Create histogram representation of data.
        2. Initialization: Set up initial centroids with quantile-based placement.
        3. Zero handling: If data includes negative values, ensure zero is a centroid.
        4. Iteration loop:
           a. Assign histogram bins to nearest centroid.
           b. Update each centroid as weighted average of assigned bins.
           c. Restore fixed centroids to their original positions.
           d. Save variant every 'saving_intervals' iterations.
           e. Check convergence (centroid change < 0.00001).
        5. Termination: Stop at max_iter or when centroids stabilize.

        :param X_train: Training data tensor to cluster.
        :param importance: Importance weight for each data point in X_train.
        :param init: Initial centroid positions, with at least first and last positions set.
        :param fixed: List of indices for centroids that should not move during optimization.
            Default: [0, n_clusters//2, n_clusters-1] for data with negative values,
            [0, n_clusters-1] for non-negative data.
        :param intervals: Number of histogram bins to use for efficient updates, default 700.
        """
        if self.max_iter == 1:
            self.centroids = deepcopy(init)
            return
        if fixed is None:
            fixed = [0, len(init) // 2, len(init) - 1]

        self.hist = KMeansWeighted.create_histogramm_sorted(X_train, importance, intervals=intervals)

        init_by_hist = self.get_init(self.hist.centroids, self.hist.weighted_importance, self.n_clusters)
        init_by_hist[0] = init[0]
        init_by_hist[-1] = init[-1]
        zero_idx = fns.argmin(fns.abs(init_by_hist[:]), axis=0).item()

        if init[0] <= 0.0:
            init_by_hist[zero_idx] = 0.0  # to have zero in codebook
            fixed[1] = zero_idx
        init = init_by_hist

        self.centroids = deepcopy(init)

        # not only last variant is stored,
        # but also intermediate ones for choosing codebook which gives minimum diff in MatMul
        saving_intervals = 5
        iteration = 0
        prev_centroids = self.centroids
        while iteration < self.max_iter:
            prev_centroids = deepcopy(self.centroids)

            if iteration % saving_intervals == 0:
                self.variants.append(deepcopy(self.centroids))

            centroid_idxs = round_to_left(self.centroids, self.hist.centroids)
            for i in range(self.n_clusters):
                idxs = fns.nonzero(centroid_idxs == i)
                if len(idxs[0]) == 0:
                    continue
                self.centroids[i] = (
                    fns.sum(self.hist.weighted_centroids[idxs]).item()
                    / fns.sum(self.hist.weighted_importance[idxs]).item()
                )

            for idx in fixed:
                self.centroids[idx] = init[idx]
            iteration += 1
            if fns.any(fns.all(fns.abs(self.centroids - prev_centroids) < 0.00001)):
                break

        self.variants.append(deepcopy(self.centroids))

    def evaluate(self, X) -> tuple[Tensor, Tensor]:
        """
        Assign data points to their nearest centroids.

        :param X: Input data tensor to assign to clusters.
        :return: Tuple of (centroids, assignments) where centroids is a flattened tensor
            of centroid values and assignments is a tensor of cluster indices for each input value.
        """
        centroid_idxs = round_to_left(self.centroids, X)
        return deepcopy(self.centroids).flatten(), centroid_idxs


def weights_clusterization_k_means(
    weight, importance, n_centroids=2**4, intervals=700, max_iter=70
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """
    Perform weighted K-means clustering on weight data to find optimal codebook.

    This is the main entry point for clustering weight values into a codebook of
    discrete values suitable for quantization. It applies weighted K-means with
    fixed boundary centroids to ensure proper coverage of the weight distribution.

    Algorithm:
    1. Flatten weight and importance tensors for 1D clustering.
    2. Initialize min and max centroids from data range.
    3. Perform weighted K-means with fixed min, zero (if applicable), and max centroids.
    4. Return codebook (centroids), index assignments, and intermediate variants.

    :param weight: Weight tensor to cluster, will be flattened internally.
    :param importance: Importance weights for each element, typically derived from
        activation statistics. Same shape as weight.
    :param n_centroids: Number of codebook entries (clusters), default 16 for 4-bit quantization.
    :param intervals: Number of histogram bins for efficient clustering, default 700.
    :param max_iter: Maximum number of K-means iterations, default 70.
    :return: Tuple of (codebook, indexes, variants) where codebook is a tensor of optimal
        centroid values with shape (n_centroids,), indexes is a tensor of cluster assignments
        reshaped to original weight shape, and variants is a list of alternative codebook
        configurations from different iterations.
    """
    orig_shape = weight.shape
    weight = weight.flatten()
    importance = importance.flatten()

    n_init = [0, 0]
    n_init[0] = weight.min()
    n_init[-1] = weight.max()

    kmeans = KMeansWeighted(n_centroids, max_iter=max_iter)

    # fixed centroids: min, zero, max
    kmeans.fit(
        weight,
        importance,
        n_init,
        fixed=[0, n_centroids // 2 - 1, n_centroids - 1] if n_init[0] < 0.0 else [0, n_centroids - 1],
        intervals=intervals,
    )
    codebook, indexes = kmeans.evaluate(weight)

    indexes = fns.reshape(indexes, orig_shape)

    return codebook, indexes, kmeans.variants
