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


def add_visual_pruning_args(parser):
    group = parser.add_argument_group("Visual Token Pruning Arguments")
    group.add_argument("--enable_visual_pruning", action="store_true", help="Enable visual token pruning")
    group.add_argument("--num_keep_tokens", type=int, default=128, help="Number of visual tokens to keep")
    group.add_argument("--theta", type=float, default=0.5, help="Balance factor for diversity vs relevance")
    return parser
