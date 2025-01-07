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
import json


def check_serialization(obj, comparator=None):
    state = obj.get_state()

    serialized_state = json.dumps(state, sort_keys=True, indent=4)
    deserialized_state = json.loads(serialized_state)

    obj_from_state = obj.from_state(state)
    obj_from_deserialized_state = obj.from_state(deserialized_state)
    if comparator:
        assert comparator(obj, obj_from_state)
        assert comparator(obj, obj_from_deserialized_state)
    else:
        assert obj == obj_from_state
        assert obj == obj_from_deserialized_state
