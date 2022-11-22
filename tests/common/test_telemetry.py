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
from nncf.telemetry import TelemetryExtractor
from nncf.telemetry import tracked_function
from nncf.telemetry.extractors import CollectedEvent


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


CATEGORY_FOR_TEST = "test_category"
NAME_OF_EVENT_FOR_TEST = "test_category"
CONSTANT_DEFAULT_ARGVALUE = "foo"
EVENT_INT_DATA = 665


def test_tracked_function(mocker):
    from nncf.telemetry import NNCFTelemetry
    send_event_spy = mocker.spy(NNCFTelemetry, "send_event")
    start_session_event_spy = mocker.spy(NNCFTelemetry, "start_session")
    end_session_event_spy = mocker.spy(NNCFTelemetry, "end_session")

    class DoubleArgExtractor(TelemetryExtractor):
        def extract(self, argvalue: str) -> CollectedEvent:
            val = argvalue + argvalue
            return CollectedEvent(name=NAME_OF_EVENT_FOR_TEST,
                                  data=val,
                                  int_data=EVENT_INT_DATA)

    class NoArgExtractor(TelemetryExtractor):
        def extract(self, argvalue) -> CollectedEvent:
            return CollectedEvent(name=NAME_OF_EVENT_FOR_TEST)

    @tracked_function(category=CATEGORY_FOR_TEST,
                      collectors=["arg2", NoArgExtractor("arg1"), "arg3", DoubleArgExtractor("arg2")])
    def fn_to_test(arg1, arg2, arg3 = CONSTANT_DEFAULT_ARGVALUE):
        pass

    fn_to_test("bar", "baz", "qux")
    assert start_session_event_spy.call_count == 4
    assert end_session_event_spy.call_count == 4
    assert send_event_spy == 4

    expected_call_args_list = [
        (CATEGORY_FOR_TEST, NAME_OF_EVENT_FOR_TEST, "baz"),
        (CATEGORY_FOR_TEST, NAME_OF_EVENT_FOR_TEST),
        (CATEGORY_FOR_TEST, NAME_OF_EVENT_FOR_TEST, "qux"),
        (CATEGORY_FOR_TEST, NAME_OF_EVENT_FOR_TEST, "bazbaz")]

    assert send_event_spy.call_args_list == expected_call_args_list
