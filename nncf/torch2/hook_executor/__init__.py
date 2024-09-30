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

from nncf.torch2.hook_executor.wrapper import get_hook_storage as get_hook_storage
from nncf.torch2.hook_executor.wrapper import insert_hook as insert_hook
from nncf.torch2.hook_executor.wrapper import is_wrapped as is_wrapped
from nncf.torch2.hook_executor.wrapper import remove_group as remove_group
from nncf.torch2.hook_executor.wrapper import wrap_model as wrap_model
