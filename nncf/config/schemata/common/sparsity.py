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
from nncf.config.schemata.basic import ARRAY_OF_NUMBERS
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.defaults import RB_SPARSITY_SCHEDULER
from nncf.config.schemata.defaults import SPARSITY_FREEZE_EPOCH
from nncf.config.schemata.defaults import SPARSITY_LEVEL_SETTING_MODE
from nncf.config.schemata.defaults import SPARSITY_MULTISTEP_SPARSITY_LEVELS
from nncf.config.schemata.defaults import SPARSITY_MULTISTEP_STEPS
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_CONCAVE
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_PATIENCE
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_POWER
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_UPDATE_PER_OPTIMIZER_STEP
from nncf.config.schemata.defaults import SPARSITY_TARGET
from nncf.config.schemata.defaults import SPARSITY_TARGET_EPOCH

SCHEDULE_OPTIONS = [
    "polynomial",
    "exponential",
    "adaptive",
    "multistep",
]

COMMON_SPARSITY_PARAM_PROPERTIES = {
    "sparsity_level_setting_mode": with_attributes(
        STRING,
        description="The mode of sparsity level setting. \n"
        "`global` - the sparsity level is calculated across all "
        "weight values in the network across layers, "
        "`local` - the sparsity level can be set per-layer and "
        "within each layer is computed with respect only to the "
        "weight values within that layer.",
        default=SPARSITY_LEVEL_SETTING_MODE,
    ),
    "schedule": with_attributes(
        STRING,
        description=f"The type of scheduling to use for adjusting the target"
        f"sparsity level. Default - {RB_SPARSITY_SCHEDULER} for `rb_sparsity`, "
        f"{SPARSITY_SCHEDULER} otherwise",
        enum=SCHEDULE_OPTIONS,
    ),
    "sparsity_target": with_attributes(
        NUMBER,
        description="Target sparsity level for the model, to be reached at the end of the compression schedule.",
        default=SPARSITY_TARGET,
    ),
    "sparsity_target_epoch": with_attributes(
        NUMBER,
        description="Index of the epoch upon which the sparsity level "
        "of the model is scheduled to become "
        "equal to `sparsity_target`.",
        default=SPARSITY_TARGET_EPOCH,
    ),
    "sparsity_freeze_epoch": with_attributes(
        NUMBER,
        description="Index of the epoch upon which the sparsity mask will be frozen and no longer trained.",
        default=SPARSITY_FREEZE_EPOCH,
    ),
    "update_per_optimizer_step": with_attributes(
        BOOLEAN,
        description="Whether the function-based sparsity level schedulers "
        "should update the sparsity level after each optimizer "
        "step instead of each epoch step.",
        default=SPARSITY_SCHEDULER_UPDATE_PER_OPTIMIZER_STEP,
    ),
    "steps_per_epoch": with_attributes(
        NUMBER,
        description="Number of optimizer steps in one epoch. Required to start proper "
        "scheduling in the first training epoch if "
        "`update_per_optimizer_step` is `true.`",
    ),
    "multistep_steps": with_attributes(
        ARRAY_OF_NUMBERS,
        description="A list of scheduler steps at which to transition "
        "to the next scheduled sparsity level (multistep "
        "scheduler only).",
        default=SPARSITY_MULTISTEP_STEPS,
    ),
    "multistep_sparsity_levels": with_attributes(
        ARRAY_OF_NUMBERS,
        description="Multistep scheduler only - Levels of sparsity to use at "
        "each step of the scheduler as specified in the "
        "`multistep_steps` attribute. The first sparsity level "
        "will be applied immediately, "
        "so the length of this list should be larger than the "
        "length of the `multistep_steps` by one. The last "
        "sparsity level will function as the ultimate sparsity "
        'target, overriding the "sparsity_target" setting if it '
        "is present.",
        default=SPARSITY_MULTISTEP_SPARSITY_LEVELS,
    ),
    "patience": with_attributes(
        NUMBER,
        description="A conventional patience parameter for the scheduler, "
        "as for any other standard scheduler. Specified in units "
        "of scheduler steps.",
        default=SPARSITY_SCHEDULER_PATIENCE,
    ),
    "power": with_attributes(
        NUMBER,
        description="For polynomial scheduler - determines the corresponding power value.",
        default=SPARSITY_SCHEDULER_POWER,
    ),
    "concave": with_attributes(
        BOOLEAN,
        description="For polynomial scheduler - if `true`, then the target sparsity "
        "level will be approached in concave manner, and in convex "
        "manner otherwise.",
        default=SPARSITY_SCHEDULER_CONCAVE,
    ),
}
