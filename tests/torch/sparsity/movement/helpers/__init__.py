"""
 Copyright (c) 2022 Intel Corporation
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

# TODO(yujie): in the mid of refactoring this helper. temporal status
from tests.torch.sparsity.movement.helpers.config import NNCFAlgoConfig
from tests.torch.sparsity.movement.helpers.config import SchedulerParams
from tests.torch.sparsity.movement.helpers.run_recipe import BaseMockRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import BertRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import Conv2dPlusLinearRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import Conv2dRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import LinearRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import SwinRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import TransformerBlockItemOrderedDict
from tests.torch.sparsity.movement.helpers.run_recipe import Wav2Vec2RunRecipe
from tests.torch.sparsity.movement.helpers.trainer import CompressionCallback
from tests.torch.sparsity.movement.helpers.trainer import CompressionTrainer
from tests.torch.sparsity.movement.helpers.trainer import build_compression_trainer
from tests.torch.sparsity.movement.helpers.utils import force_update_sparsifier_binary_masks_by_threshold
from tests.torch.sparsity.movement.helpers.utils import initialize_sparsifier_parameters_by_linspace
from tests.torch.sparsity.movement.helpers.utils import is_roughly_non_decreasing
from tests.torch.sparsity.movement.helpers.utils import is_roughly_of_same_value
from tests.torch.sparsity.movement.helpers.utils import mock_linear_nncf_node
