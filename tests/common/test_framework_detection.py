# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
import sys
from importlib import import_module
from importlib.machinery import ModuleSpec
from typing import List
from unittest import mock
from unittest.mock import MagicMock

import pytest

import nncf

SUPPORTED_FRAMEWORKS = nncf._SUPPORTED_FRAMEWORKS  # pylint:disable=protected-access


@pytest.mark.parametrize("ref_available_frameworks", [["torch"], ["torch", "tensorflow"], ["onnx", "openvino"], []])
def test_frameworks_detected(ref_available_frameworks: List[str], nncf_caplog):
    with mock.patch.dict(sys.modules):
        for supp_fw in SUPPORTED_FRAMEWORKS:
            if supp_fw in sys.modules:
                del sys.modules[supp_fw]
        del sys.modules["nncf"]

        for fw in ref_available_frameworks:
            mock_spec = ModuleSpec(fw, loader=MagicMock(), origin="foo/bar")
            module = MagicMock()
            module.__spec__ = mock_spec
            sys.modules[fw] = module

        with nncf_caplog.at_level(logging.INFO):
            import_module("nncf")
            matches = re.search(r"Supported frameworks detected: (.*)", nncf_caplog.text)
            if ref_available_frameworks:
                assert matches is not None
                match_text = matches[0]
                for fw in ref_available_frameworks:
                    assert fw in match_text
                unavailable_frameworks = [fw for fw in SUPPORTED_FRAMEWORKS if fw not in ref_available_frameworks]
                for fw in unavailable_frameworks:
                    assert fw not in match_text
            else:
                assert matches is None
