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

from tests.torch.sparsity.movement.helpers.config import MovementAlgoConfig
from tests.torch.sparsity.movement.helpers.run_recipe import BaseMockRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import BertRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import ClipVisionRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import Conv2dPlusLinearRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import Conv2dRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import DictInTransformerBlockOrder
from tests.torch.sparsity.movement.helpers.run_recipe import DistilBertRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import LinearRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import MobileBertRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import SwinRunRecipe
from tests.torch.sparsity.movement.helpers.run_recipe import Wav2Vec2RunRecipe
from tests.torch.sparsity.movement.helpers.trainer import CompressionCallback
from tests.torch.sparsity.movement.helpers.trainer import CompressionTrainer
from tests.torch.sparsity.movement.helpers.trainer import build_compression_trainer
from tests.torch.sparsity.movement.helpers.utils import FACTOR_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers.utils import LINEAR_LAYER_SPARSITY_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers.utils import MODEL_SPARSITY_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers.utils import MRPC_CONFIG_FILE_NAME
from tests.torch.sparsity.movement.helpers.utils import THRESHOLD_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers.utils import TRAINING_SCRIPTS_PATH
from tests.torch.sparsity.movement.helpers.utils import force_update_sparsifier_binary_masks_by_threshold
from tests.torch.sparsity.movement.helpers.utils import initialize_sparsifier_parameters_by_linspace
from tests.torch.sparsity.movement.helpers.utils import is_roughly_non_decreasing
from tests.torch.sparsity.movement.helpers.utils import is_roughly_of_same_value
from tests.torch.sparsity.movement.helpers.utils import mock_linear_nncf_node
