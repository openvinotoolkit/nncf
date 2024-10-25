# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import nncf
from nncf.api.compression import TModel
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset


def cache_weight_compression_statistics(
    model: TModel, dataset: Dataset, subset_size: int, statistics_dir_path: str
) -> None:
    """Caches compression statistics for the given model and dataset for WeightCompression."""
    backend = get_backend(model)
    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.cache_statistics import cache_weight_compression_statistics

        return cache_weight_compression_statistics(model, dataset, subset_size, statistics_dir_path)
    raise nncf.UnsupportedBackendError(f"Unsupported type of backend: {backend}")
