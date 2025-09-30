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
from dataclasses import dataclass
from typing import Optional, TypeVar
import numpy as np
import time

import openvino as ov
from openvino.runtime import opset13 as opset

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

from nncf.quantization.algorithms.weight_compression.constants import CB4_QUANTILES

TModel = TypeVar("TModel")



def fp8_convert(in_shape):
    input = opset.parameter(
        in_shape, dtype=ov.Type.f32
    )
    scale_convert = opset.convert(input, ov.Type.f8e4m3)
    scale_convert = opset.convert(scale_convert, ov.Type.f32)
    result = opset.result(scale_convert, name="Result")
    result.get_output_tensor(0).set_names(set(["Result"]))
    model = ov.Model([result], [input])

    compiled_model = ov.compile_model(model)

    return compiled_model


class CodebookEstimation:
    """
    Codebook estimation algorithm implementation.
    """

    def __init__(
        self,
    ):
        """
        Initializes the CodebookEstimation algorithm.
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
        debug=False
    ) -> dict[str, CompressedWeight]:
        """
        Estimates better codebook.
        Minimizes difference between floating point MatMul and
        MatMul with compressed weights.
        The algorithm computes codebook and indexes for MatMul compression.

        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param statistic_points: Statistic points with collected statistics values.
        :param dataset: A representative dataset for the calibration process.
        :param backend_entity: Weight compression algorithm backend.
        :return: A dictionary that maps weight names to CompressedWeight with codebook, codebook indexes and scale.
        """
        self._backend_entity = backend_entity
        if self._backend_entity is None:
            self._set_backend_entity(model)
        res = dict()

        for wp in track(all_weight_params, description="Applying Codebook Estimation"):
            weight_name = wp.weight_name
            node_name = wp.node_with_weight.node_name
            config = wp.compression_config

            if config.num_bits != 4:# or node_name not in statistics:
                res[weight_name] = CompressedWeight()
                continue

            stats = statistics[node_name]

            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            _, weight_port_id = weight_data[0]

            weight = self._backend_entity.get_weight(wp.node_with_weight, weight_port_id, model, graph)

            if debug:
                qw = float_quantize_dequantize_weight(weight, config, wp.reduction_axes)
                print("Initial diff:", fns.mean(fns.abs(weight.data - qw.data)))

            codebook, scale, indexes = self.calculate_codebook(
                stats,
                weight,
                wp.reduction_axes,
                config,
                wp
            )
            res[weight_name] = CompressedWeight(indexes, scale, None, codebook)
            config.codebook_values = codebook
  
            if debug:
                qw = float_quantize_dequantize_weight(weight, config, wp.reduction_axes)
                print("kmeans diff:", fns.mean(fns.abs(weight.data - qw.data)))

        return res

    @staticmethod
    def calculate_codebook(
        statistics: WCTensorStatistic,
        weight: Tensor,
        reduction_axes: tuple[int, ...],
        config: WeightCompressionConfig,
        wp: WeightCompressionParameters
    ) -> Tensor:

        reduction_axis = reduction_axes[0]
        weight = deepcopy(weight.astype(TensorDataType.float32))
        
        s, X = process_stats(statistics, 128)

        if reduction_axis == 0:
            weight = fns.transpose(weight)
            reduction_axis = 1

        if config.group_size != -1:
            weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)
        
        orig_shape = weight.shape
        
        importance = fns.ones_like(weight)
        importance = importance * s
        
        scale = calculate_float_quantization_params(weight, reduction_axes, config, signed=True)
        norm_weight = _calculate_normalized_weight(weight, scale)

        codebook, indexes, variants = weights_clusterization_k_means(norm_weight, importance)

        converter = fp8_convert(codebook.shape)
        indexes = indexes.reshape(orig_shape)

        
        best_codebook = converter(codebook.as_openvino_tensor().data)[0]
        
        fp_outs = fns.matmul(weight, X)
        diff = float('inf')
        
        variants[0] = fns.tensor(CB4_QUANTILES, backend=weight.backend, dtype=weight.dtype)
        variants[1] = fns.tensor([i for i in range(-8, 8)], backend=weight.backend, dtype=weight.dtype)
        best_i = -1
        
        for i_var, var in enumerate(variants):
            var = converter(var.as_openvino_tensor().data)[0]
            config.codebook_values = Tensor(var)
            qw = float_quantize_dequantize_weight(weight, config, wp.reduction_axes)
            q_outs = fns.matmul(qw, X)
            
            cur_diff = fns.mean(fns.abs(fp_outs - q_outs)).item()
            if cur_diff < diff:
                diff = cur_diff
                best_codebook = var
                best_i = i_var
            
        print("Best codebook:", best_codebook, "diff:", diff, "best_i:", best_i)

        return Tensor(best_codebook), None, None


def round(quantiles, values):
    center_of_quantiles = 0.5 * (quantiles[1:] + quantiles[:-1])
    return fns.searchsorted(center_of_quantiles, values, side='left', sorter=None)

@dataclass
class KMeansAlgoData:
    centroids: Tensor
    hist: Tensor
    weighted_hist: Tensor | None = None

    frequencies: Tensor | None = None
    weights: Tensor | None = None

class KMeansWeighted:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.variants = []

    @staticmethod
    def get_init(values, frequencies, n_clusters):
        step = 1.0 / (n_clusters - 1)
        denum = fns.sum(frequencies)
        quants = [i * step for i in range(n_clusters)]
        n_frequencies = frequencies / denum
        n_frequencies = fns.cumsum(n_frequencies)


        res = fns.zeros((n_clusters,), backend=values.backend, dtype=values.dtype)
        for i in range(len(quants)):
            if i == 0:
                res[i] = values[0]
            elif i == len(quants) - 1:
                res[i] = values[-1]
            else:
                prev = values[fns.nonzero(n_frequencies <= quants[i])[0][-1].item()].item()
                next_ = values[fns.nonzero(n_frequencies <= quants[i + 1])[0][-1].item()].item()
                res[i] = (prev + next_) / 2

        return res

    @staticmethod
    def create_histogramm(data, granularity=0.01):
        centers = []
        step = granularity

        data_range=(data.min().item(), data.max().item())
        prev = data_range[0]

        while prev < data_range[1]:
            centers.append(prev + step / 2)
            prev += step

        centers = fns.tensor(centers)
        centroid_idxs = round(centers, data)

        res = [[], [], []]
        for i in range(centers.size):
            idxs = fns.nonzero(centroid_idxs == i)
            if len(idxs[0]) == 0:
                continue
            res[0].append(centers[i])
            res[1].append(fns.sum(data[idxs]))
            res[2].append(len(idxs[0]))

        res[0] = fns.tensor(res[0]) # centers of histogram bins
        res[1] = fns.tensor(res[1]) # sum of values in each bin
        res[2] = fns.tensor(res[2]) # count of values in each bin

        return res

    @staticmethod
    def add_weighted_data_and_weights(res, data, importance):
        res[1].append(fns.sum(fns.multiply(data, importance)).item())
        res[2].append(fns.sum(importance).item())

    @staticmethod
    def create_histogramm_sorted(data_, importance, granularity=0.01):
        centers = []
        ranges = []
        step = data_.max().item() * granularity / 3.5

        sorted_idx = fns.argsort(data_)
        data = data_[sorted_idx]
        importance = importance[sorted_idx]
        
        #data = np.array([data_, importance])
        #data = data[:, data[0, :].argsort()]

        data_range=(data.min().item(), data.max().item())
        prev = data_range[0]

        
        while prev < data_range[1]:
            centers.append(prev + step / 2)
            prev += step
            
            if len(centers) > 1:
                ranges.append(0.5 * (centers[-2] + centers[-1]))
            ranges.append(centers[-1])


        centers = fns.tensor(centers, backend=data_.backend, dtype=data_.dtype)
        ranges = fns.tensor(ranges, backend=data_.backend, dtype=data_.dtype)

        ranges_idxs = round(data, ranges)

        res = [[], [], []]
        for i in range(centers.size):
            res[0].append(centers[i])
            if i == 0:
                KMeansWeighted.add_weighted_data_and_weights(res, data[:ranges_idxs[1].item()], importance[:ranges_idxs[1].item()])
            elif i == centers.size - 1:
                KMeansWeighted.add_weighted_data_and_weights(res, data[ranges_idxs[-2].item():], importance[ranges_idxs[-2].item():])
            else:
                idx = 2 * i
                KMeansWeighted.add_weighted_data_and_weights(res, data[ranges_idxs[idx - 1].item():ranges_idxs[idx + 1].item()],
                                                             importance[ranges_idxs[idx - 1].item():ranges_idxs[idx + 1].item()])

        res[0] = centers #fns.tensor(res[0], backend=data_.backend, dtype=data_.dtype) # centers of histogram bins
        res[1] = fns.tensor(res[1], backend=data_.backend, dtype=data_.dtype)
        res[2] = fns.tensor(res[2], backend=data_.backend, dtype=data_.dtype)

        return res

    def fit(self, X_train, importance, init, fixed=[]):
        if self.max_iter == 1:
            self.centroids = deepcopy(init)
            return
        
        self.hist = KMeansWeighted.create_histogramm_sorted(X_train, importance)

        init_by_hist = self.get_init(self.hist[0], self.hist[2], self.n_clusters)
        init_by_hist[0] = init[0]
        init_by_hist[-1] = init[-1]
        zero_idx = fns.argmin(fns.abs(init_by_hist[:]))
        init_by_hist[zero_idx] = 0.0 #init[0, zero_idx]
        fixed[1] = zero_idx
        init = init_by_hist

        self.centroids = deepcopy(init)

        iteration = 0
        prev_centroids = self.centroids
        while iteration < self.max_iter:
            prev_centroids = deepcopy(self.centroids)
            
            if iteration % 5 == 0:
                self.variants.append(deepcopy(self.centroids))

            centroid_idxs = round(self.centroids, self.hist[0])
            for i in range(self.n_clusters):
                idxs = fns.nonzero(centroid_idxs == i)
                self.centroids[i] = fns.sum(self.hist[1][idxs]).item() / fns.sum(self.hist[2][idxs]).item()

            # for i, centroid in enumerate(self.centroids):
            #     if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
            #         self.centroids[i] = prev_centroids[i]
            for idx in fixed:
                self.centroids[idx] = init[idx]
            iteration += 1
            if fns.any(fns.all(fns.abs(self.centroids - prev_centroids) < 0.00001)):
                break
            # if np.all(np.abs(self.centroids - prev_centroids) < 0.00001).any():
            #     break
        
        self.variants.append(deepcopy(self.centroids))


    def evaluate(self, X):
        centroid_idxs = round(self.centroids, X)
        return deepcopy(self.centroids).flatten(), centroid_idxs


def weights_clusterization_k_means(weight, importance, n_centroids=2**4):
    #weight = weight.as_numpy_tensor().data
    #importance = importance.as_numpy_tensor().data

    ow = deepcopy(weight)
    orig_shape = weight.shape
    weight = weight.flatten()
    importance = importance.flatten()

    n_init = [0, 0]
    n_init[0] = weight.min()
    n_init[-1] = weight.max()

    kmeans = KMeansWeighted(n_centroids, max_iter=70)

    kmeans.fit(weight, importance, n_init, fixed=[0, 7, 15])
    codebook, indexes = kmeans.evaluate(weight)#.reshape(-1, 1))

    indexes = fns.reshape(indexes, orig_shape)

    #print(orig_shape, np.mean(np.abs(ow - codebook[indexes])))
    
    print(orig_shape, fns.mean(fns.abs(ow - codebook[indexes])))

    return codebook, indexes, kmeans.variants
