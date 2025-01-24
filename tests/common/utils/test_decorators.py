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
from typing import List, Optional, Type
from unittest.mock import patch

import pytest

from nncf.common.utils.decorators import IMPORTED_DEPENDENCIES
from nncf.common.utils.decorators import skip_if_dependency_unavailable


@dataclass
class StructForTest:
    dependencies: List[str]
    side_effect: Optional[Type[Exception]]
    result_is_none: bool
    final_imported_dependencies: bool


@pytest.mark.parametrize(
    "ts",
    [
        StructForTest(
            dependencies=["dependency1", "dependency2"],
            side_effect=None,
            result_is_none=False,
            final_imported_dependencies=True,
        ),
        StructForTest(
            dependencies=["nonexistent_dependency"],
            side_effect=ImportError("Module not found"),
            result_is_none=True,
            final_imported_dependencies=False,
        ),
    ],
)
def test_case_parametrized(ts: StructForTest, mocker):
    mocked_func = mocker.MagicMock()
    wrapped_func = skip_if_dependency_unavailable(ts.dependencies)(mocked_func)

    with patch("nncf.common.utils.decorators.import_module", side_effect=ts.side_effect):
        wrapped_func()

    result = wrapped_func()

    if ts.result_is_none:
        assert result is None
        mocked_func.assert_not_called()
    else:
        mocked_func.assert_called_once

    for d in IMPORTED_DEPENDENCIES:
        assert IMPORTED_DEPENDENCIES[d] == ts.final_imported_dependencies

    IMPORTED_DEPENDENCIES.clear()
