
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
import os
import importlib
from unittest import mock
from unittest.mock import MagicMock

import pytest

from nncf.definitions import NNCF_CI_ENV_VAR_NAME
from nncf.definitions import NNCF_DEV_ENV_VAR_NAME

@pytest.mark.parametrize("env_var_to_define", [NNCF_CI_ENV_VAR_NAME,
                                               NNCF_DEV_ENV_VAR_NAME])
def test_telemetry_is_mocked_if_env_vars_defined(mocker, env_var_to_define):
    with mock.patch.dict(os.environ, {env_var_to_define: "1"}):
        # Need to reload the module where the logic concerning
        # env vars and telemetry object is evaluated
        from nncf.telemetry import wrapper
        importlib.reload(wrapper)

        # telemetry alias will no longer be available after reload,
        # so importing via a full name
        from nncf.telemetry.wrapper import NNCFTelemetry
        assert isinstance(NNCFTelemetry, MagicMock)
    # cleanup
    importlib.reload(wrapper)
