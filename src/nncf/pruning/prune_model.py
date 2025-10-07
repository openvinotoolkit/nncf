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

from typing import Any, Optional

import nncf
from nncf.api.compression import TModel
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import PruneMode
from nncf.scopes import IgnoredScope


def prune(
    model: TModel,
    *,
    mode: PruneMode,
    ratio: float,
    ignored_scope: Optional[IgnoredScope] = None,
    examples_inputs: Optional[Any] = None,
) -> TModel:
    """
    Prunes the given model based on the specified mode and ratio.
    Insert binary masks for the parameters and calculate the initial masks by the given ratio.

    :param model: The model to be pruned.
    :param mode: The mode of pruning to be applied.
    :param ratio: The ratio of parameters to prune from the model.
    :param ignored_scope: A scope of parameters to ignore during pruning. Defaults to None.
    :param examples_inputs: Example inputs to guide the pruning process. Defaults to None.
    :return: The pruned model.
    """
    backend = get_backend(model)
    if backend == BackendType.TORCH:
        from nncf.torch.function_hook.prune.prune_model import prune

        model = prune(model, mode, ratio, ignored_scope, examples_inputs)
    else:
        msg = f"Pruning is not supported for the {backend} backend."
        raise nncf.InternalError(msg)
    return model
