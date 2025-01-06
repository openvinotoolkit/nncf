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

import nncf


def get_built_model(model, config):
    if not model.built:
        input_info = config.get("input_info", {})
        if isinstance(input_info, dict):
            sample_size = input_info.get("sample_size", None)
        else:
            sample_size = input_info[0].get("sample_size", None) if input_info else None
        if not sample_size:
            raise nncf.ValidationError("sample_size must be provided in configuration file")
        model.build([None] + list(sample_size[1:]))

    return model
