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

from nncf.experimental.torch2.function_hook.graph.build_graph_mode import build_graph as build_graph
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage as get_hook_storage
from nncf.experimental.torch2.function_hook.wrapper import is_wrapped as is_wrapped
from nncf.experimental.torch2.function_hook.wrapper import register_post_function_hook as register_post_function_hook
from nncf.experimental.torch2.function_hook.wrapper import register_pre_function_hook as register_pre_function_hook
from nncf.experimental.torch2.function_hook.wrapper import wrap_model as wrap_model
