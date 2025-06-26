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

from dataclasses import dataclass


@dataclass
class PatchingState:
    """
    A class to track which pytorch components were patched by NNCF.
    """

    jit_is_wrapped: bool = False
    operators_are_wrapped: bool = False
    compile_is_wrapped: bool = False


PATCHING_STATE = PatchingState()
