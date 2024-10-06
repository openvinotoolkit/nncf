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


from dataclasses import dataclass
from typing import Any, Optional

import pytest

from nncf.torch2.hook_executor.hook_executor_mode import generate_normalized_op_name


@dataclass
class NormalizedOpNameTestCase:
    model_name: str
    fn_name: str
    call_id: Optional[str]
    ref: str


def idfn(value: Any):
    if isinstance(value, NormalizedOpNameTestCase):
        return value.ref
    return None


@pytest.mark.parametrize(
    "param",
    (
        NormalizedOpNameTestCase("module", "foo", None, "module/foo"),
        NormalizedOpNameTestCase("module", "foo", 0, "module/foo/0"),
        NormalizedOpNameTestCase("module", "foo", 1, "module/foo/1"),
        NormalizedOpNameTestCase("module.module", "foo", None, "module:module/foo"),
        NormalizedOpNameTestCase("module.module", "foo", 0, "module:module/foo/0"),
        NormalizedOpNameTestCase("hook.module:module/mod", "foo", None, "hook:module:module-mod/foo"),
    ),
    ids=idfn,
)
def test_generate_normalized_op_name(param: NormalizedOpNameTestCase):
    op_name = generate_normalized_op_name(module_name=param.model_name, fn_name=param.fn_name, call_id=param.call_id)
    assert op_name == param.ref
