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
from nncf.config.schemata.basic import ARRAY_OF_NUMBERS
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes

COMMON_SPARSITY_PARAM_PROPERTIES = {
    "schedule": with_attributes(STRING,
                                description="The type of scheduling to use for adjusting the target"
                                            "sparsity level"),
    "patience": with_attributes(NUMBER,
                                description="A regular patience parameter for the scheduler, "
                                            "as for any other standard scheduler. Specified in units "
                                            "of scheduler steps."),
    "power": with_attributes(NUMBER,
                             description="For polynomial scheduler - determines the corresponding power value."),
    "concave": with_attributes(BOOLEAN, description="For polynomial scheduler - if True, then the target sparsity "
                                                    "level will be approached in concave manner, and in convex "
                                                    "manner otherwise."),
    "sparsity_target": with_attributes(NUMBER,
                                       description="Target value of the sparsity level for the model"),
    "sparsity_target_epoch": with_attributes(NUMBER,
                                             description="Index of the epoch from which the sparsity level "
                                                         "of the model will be equal to spatsity_target value"),
    "sparsity_freeze_epoch": with_attributes(NUMBER,
                                             description="Index of the epoch from which the sparsity mask will "
                                                         "be frozen and no longer trained"),
    "update_per_optimizer_step": with_attributes(BOOLEAN,
                                                 description="Whether the function-based sparsity level schedulers "
                                                             "should update the sparsity level after each optimizer "
                                                             "step instead of each epoch step."),
    "steps_per_epoch": with_attributes(NUMBER,
                                       description="Number of optimizer steps in one epoch. Required to start proper "
                                                   " scheduling in the first training epoch if "
                                                   "'update_per_optimizer_step' is true"),
    "multistep_steps": with_attributes(ARRAY_OF_NUMBERS,
                                       description="A list of scheduler steps at which to transition "
                                                   "to the next scheduled sparsity level (multistep "
                                                   "scheduler only)."),
    "multistep_sparsity_levels": with_attributes(ARRAY_OF_NUMBERS,
                                                 description="Levels of sparsity to use at each step of the scheduler "
                                                             "as specified in the 'multistep_steps' attribute. The "
                                                             "first sparsity level will be applied immediately, "
                                                             "so the length of this list should be larger than the "
                                                             "length of the 'steps' by one. The last sparsity level "
                                                             "will function as the ultimate sparsity target, "
                                                             "overriding the \"sparsity_target\" setting if it is "
                                                             "present."),
    "sparsity_level_setting_mode": with_attributes(STRING,
                                                   description="The mode of sparsity level setting( "
                                                               "'global' - one sparsity level is set for all layer, "
                                                               "'local' - sparsity level is set per-layer.)"),
}
