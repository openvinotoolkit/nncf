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

from copy import deepcopy
from typing import Optional, TypeVar
import numpy as np

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging.track_progress import track
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_float_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_float_quantization_params
from nncf.quantization.algorithms.weight_compression.weight_lowering import _calculate_normalized_weight
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


f8e4m3_data = np.array(
    [-4.4800000e+02, -4.1600000e+02, -3.8400000e+02, -3.5200000e+02
    , -3.2000000e+02, -2.8800000e+02, -2.5600000e+02, -2.4000000e+02
    , -2.2400000e+02, -2.0800000e+02, -1.9200000e+02, -1.7600000e+02
    , -1.6000000e+02, -1.4400000e+02, -1.2800000e+02, -1.2000000e+02
    , -1.1200000e+02, -1.0400000e+02, -9.6000000e+01, -8.8000000e+01
    , -8.0000000e+01, -7.2000000e+01, -6.4000000e+01, -6.0000000e+01
    , -5.6000000e+01, -5.2000000e+01, -4.8000000e+01, -4.4000000e+01
    , -4.0000000e+01, -3.6000000e+01, -3.2000000e+01, -3.0000000e+01
    , -2.8000000e+01, -2.6000000e+01, -2.4000000e+01, -2.2000000e+01
    , -2.0000000e+01, -1.8000000e+01, -1.6000000e+01, -1.5000000e+01
    , -1.4000000e+01, -1.3000000e+01, -1.2000000e+01, -1.1000000e+01
    , -1.0000000e+01, -9.0000000e+00, -8.0000000e+00, -7.5000000e+00
    , -7.0000000e+00, -6.5000000e+00, -6.0000000e+00, -5.5000000e+00
    , -5.0000000e+00, -4.5000000e+00, -4.0000000e+00, -3.7500000e+00
    , -3.5000000e+00, -3.2500000e+00, -3.0000000e+00, -2.7500000e+00
    , -2.5000000e+00, -2.2500000e+00, -2.0000000e+00, -1.8750000e+00
    , -1.7500000e+00, -1.6250000e+00, -1.5000000e+00, -1.3750000e+00
    , -1.2500000e+00, -1.1250000e+00, -1.0000000e+00, -9.3750000e-01
    , -8.7500000e-01, -8.1250000e-01, -7.5000000e-01, -6.8750000e-01
    , -6.2500000e-01, -5.6250000e-01, -5.0000000e-01, -4.6875000e-01
    , -4.3750000e-01, -4.0625000e-01, -3.7500000e-01, -3.4375000e-01
    , -3.1250000e-01, -2.8125000e-01, -2.5000000e-01, -2.3437500e-01
    , -2.1875000e-01, -2.0312500e-01, -1.8750000e-01, -1.7187500e-01
    , -1.5625000e-01, -1.4062500e-01, -1.2500000e-01, -1.1718750e-01
    , -1.0937500e-01, -1.0156250e-01, -9.3750000e-02, -7.8125000e-02
    , -7.0312500e-02, -5.8593750e-02, -5.0781250e-02, -3.9062500e-02
    , -2.9296875e-02, -1.9531250e-02, -9.7656250e-03, 0.0000000e+00
    , 9.7656250e-03, 1.9531250e-02, 2.9296875e-02, 3.9062500e-02
    , 5.0781250e-02, 5.8593750e-02, 7.0312500e-02, 7.8125000e-02
    , 9.3750000e-02, 1.0156250e-01, 1.0937500e-01, 1.1718750e-01
    , 1.2500000e-01, 1.4062500e-01, 1.5625000e-01, 1.7187500e-01
    , 1.8750000e-01, 2.0312500e-01, 2.1875000e-01, 2.3437500e-01
    , 2.5000000e-01, 2.8125000e-01, 3.1250000e-01, 3.4375000e-01
    , 3.7500000e-01, 4.0625000e-01, 4.3750000e-01, 4.6875000e-01
    , 5.0000000e-01, 5.6250000e-01, 6.2500000e-01, 6.8750000e-01
    , 7.5000000e-01, 8.1250000e-01, 8.7500000e-01, 9.3750000e-01
    , 1.0000000e+00, 1.1250000e+00, 1.2500000e+00, 1.3750000e+00
    , 1.5000000e+00, 1.6250000e+00, 1.7500000e+00, 1.8750000e+00
    , 2.0000000e+00, 2.2500000e+00, 2.5000000e+00, 2.7500000e+00
    , 3.0000000e+00, 3.2500000e+00, 3.5000000e+00, 3.7500000e+00
    , 4.0000000e+00, 4.5000000e+00, 5.0000000e+00, 5.5000000e+00
    , 6.0000000e+00, 6.5000000e+00, 7.0000000e+00, 7.5000000e+00
    , 8.0000000e+00, 9.0000000e+00, 1.0000000e+01, 1.1000000e+01
    , 1.2000000e+01, 1.3000000e+01, 1.4000000e+01, 1.5000000e+01
    , 1.6000000e+01, 1.8000000e+01, 2.0000000e+01, 2.2000000e+01
    , 2.4000000e+01, 2.6000000e+01, 2.8000000e+01, 3.0000000e+01
    , 3.2000000e+01, 3.6000000e+01, 4.0000000e+01, 4.4000000e+01
    , 4.8000000e+01, 5.2000000e+01, 5.6000000e+01, 6.0000000e+01
    , 6.4000000e+01, 7.2000000e+01, 8.0000000e+01, 8.8000000e+01
    , 9.6000000e+01, 1.0400000e+02, 1.1200000e+02, 1.2000000e+02
    , 1.2800000e+02, 1.4400000e+02, 1.6000000e+02, 1.7600000e+02
    , 1.9200000e+02, 2.0800000e+02, 2.2400000e+02, 2.4000000e+02
    , 2.5600000e+02, 2.8800000e+02, 3.2000000e+02, 3.5200000e+02
    , 3.8400000e+02, 4.1600000e+02, 4.4800000e+02]
)


class CodebookEstimation:
    """
    Scale estimation algorithm implementation.
    """

    def __init__(
        self,
    ):
        """
        :param subset_size: The number of samples for scale estimation.
        :param initial_steps: The number of the steps for absmax scale rectification.
        :param scale_steps: The number of the steps for grid search scale rectification
                            from 1.0 to 1.0 - 0.05 * scale_step.
        :param weight_penalty: coefficient for penalty between fp and compressed weights. If -1 then doesn't apply.
        """
        super().__init__()

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        else:
            msg = (
                "Cannot return backend-specific Scale Estimation entity because"
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
        Estimates better scale for the int4 nodes in the model.
        Minimizes per-group difference between floating point MatMul and
        MatMul with compressed weights.
        The algorithm computes weighted scale for the group of weights in MatMul, which
        shared the same scale.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param statistic_points: Statistic points with collected statistics values.
        :param dataset: A representative dataset for the calibration process.
        :param backend_entity: Weight compression algorithm backend.
        :return: Two dictionaries for estimated scales and zero points for each weight name.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)
        res = dict()

        for wp in track(all_weight_params, description="Applying Codebook Estimation"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if config.num_bits != 4 or node_name not in statistics:
                res[weight_name] = CompressedWeight()
                continue

            stats = statistics[node_name]

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

            scale, zero_point = self.calculate_quantization_params(
                stats,
                weight,
                wp.reduction_axes,
                config,
                self._subset_size,
                self._initial_steps,
                self._scale_steps,
                self._weight_penalty,
            )
            res[weight_name] = CompressedWeight(None, scale, zero_point, None)

        return res

    @staticmethod
    def calculate_codebook(
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        subset_size: int = 32,
        initial_steps: int = 5,
        scale_steps: int = 10,
        weight_penalty: float = -1.0,
    ) -> Tensor:

        reduction_axis = reduction_axes[0]
        weight = deepcopy(weight.astype(TensorDataType.float32))

        if reduction_axis == 0:
            weight = fns.transpose(weight)
            reduction_axis = 1

        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)
        
        scale = calculate_float_quantization_params(weight, reduction_axes, config)
        norm_weight = _calculate_normalized_weight(weight, scale)
        
        codebook, indexes = weights_clusterization_k_means(norm_weight)
        
        
            
        
        fp8_scales = np.unique(np.abs(f8e4m3_data))
        fp8_scales = fp8_scales[scale >= 1.0]
        
        
        
        
        return None


def most_common(lst):
    """
    Return the most frequently occuring element in a list.
    """
    return max(set(lst), key=lst.count)


def euclidean_(point, data):
    """
    Return euclidean distances between a point & a dataset
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


def round(quantiles, values):
    center_of_quantiles = 0.5 * (quantiles[1:] + quantiles[:-1])
    
    return np.searchsorted(center_of_quantiles, values, side='left', sorter=None)


class KMeansHist:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    @staticmethod
    def get_init(values, frequencies, n_clusters):
        step = 1.0 / n_clusters
        denum = np.sum(frequencies)
        quants = [i * step for i in range(n_clusters)]
        n_frequencies = frequencies / denum
        n_frequencies = np.cumsum(n_frequencies)

        res = []
        for i in range(len(quants)):
            if i == 0:
                res.append(values[0])
            elif i == len(quants) - 1:
                res.append(values[-1])
            else:
                prev = values[np.where(n_frequencies <= quants[i])[0][-1]]
                next_ = values[np.where(n_frequencies <= quants[i + 1])[0][-1]]
                res.append((prev + next_) / 2)

        res = np.array(res).reshape(1, -1)
        return res

    @staticmethod
    def create_histogramm(data, data_range=(-1.0, 1.0), granularity=0.01):
        centers = []
        step = granularity
        prev = data_range[0]

        while prev < data_range[1]:
            centers.append(prev + step / 2)
            prev += step

        centers = np.array(centers)
        centroid_idxs = round(centers, data)

        res = [[], [], []]
        for i in range(centers.shape[1]):
            idxs = np.where(centroid_idxs == i)
            if len(idxs[0]) == 0:
                continue
            res[0].append(centers[i])
            res[1].append(np.sum(data[idxs, :]))
            res[2].append(len(idxs[0]))

        res[0] = np.array(res[0]).reshape(-1, 1)
        res[1] = np.array(res[1])
        res[2] = np.array(res[2])

        return res

    def fit(self, X_train, init, fixed=[]):
        if self.max_iter == 1:
            self.centroids = deepcopy(init)
            return

        self.hist = self.create_histogramm(X_train)

        init_by_hist = self.get_init(self.hist[0], self.hist[2], self.n_clusters)
        init_by_hist[0, 0] = init[0]
        init_by_hist[0, -1] = init[-1]
        zero_idx = np.argmin(np.abs(init_by_hist[0, :]))
        init_by_hist[0, zero_idx] = init[0, zero_idx]
        fixed[1] = zero_idx
        init = init_by_hist

        self.centroids = deepcopy(init)

        iteration = 0
        prev_centroids = self.centroids
        while iteration < self.max_iter:
            prev_centroids = deepcopy(self.centroids)

            centroid_idxs = round(self.centroids, self.hist[0])
            for i in range(self.n_clusters):
                idxs = np.where(centroid_idxs == i)
                self.centroids[:, i] = np.sum(self.hist[1][idxs]) / np.sum(self.hist[2][idxs])

            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            for idx in fixed:
                self.centroids[:, idx] = init[:, idx]
            iteration += 1
            if np.all(np.abs(self.centroids - prev_centroids) < 0.00001).any():
                break
        print(self.centroids)

    def evaluate(self, X):
        centroid_idxs = round(self.centroids, X)
        return deepcopy(self.centroids).flatten(), centroid_idxs


def weights_clusterization_k_means(weight, n_centroids=2**4):
    weight = weight.as_numpy_tensor().data

    ow = deepcopy(weight)
    orig_shape = weight.shape
    weight = weight.flatten()
    
    n_init = [0, 0]
    n_init[0] = weight.min()
    n_init[-1] = weight.max()

    kmeans = KMeansHist(n_centroids, max_iter=1)
    
    n_init = kmeans.get_init(weight, n_init, n_centroids)
    
    #kmeans.fit(weight.reshape(-1, 1), n_init.reshape(1, -1), fixed=[0, 7, 15])
    kmeans.fit(weight, n_init, fixed=[0, 7, 15])
    codebook, indexes = kmeans.evaluate(weight.reshape(-1, 1))
    # codebook = kmeans.cluster_centers_.flatten()
    # indexes  = kmeans.labels_

    indexes = np.reshape(indexes, orig_shape)

    print(orig_shape, np.mean(np.abs(ow - codebook[indexes])))

    return codebook, indexes
