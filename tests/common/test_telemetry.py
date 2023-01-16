"""
 Copyright (c) 2023 Intel Corporation
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
import sys
from unittest import mock
from unittest.mock import call
from unittest.mock import MagicMock
from typing import Tuple

import pytest

from nncf.definitions import NNCF_CI_ENV_VAR_NAME
from nncf.definitions import NNCF_DEV_ENV_VAR_NAME
from nncf.telemetry import TelemetryExtractor
from nncf.telemetry import tracked_function
from nncf.telemetry.extractors import CollectedEvent
from nncf.telemetry.wrapper import NNCFTelemetryStub

@pytest.fixture(name="hide_pytest")
def hide_pytest_(request):
    with mock.patch.dict(sys.modules):
        del sys.modules["pytest"]
        yield request


def test_telemetry_is_not_mocked_in_normal_conditions(hide_pytest):
    with mock.patch.dict(os.environ, clear=True):
        # Need to reload the module where the logic concerning
        # env vars and telemetry object is evaluated
        from nncf.telemetry import wrapper
        importlib.reload(wrapper)

        # telemetry alias will no longer be available after reload,
        # so importing via a full name
        from nncf.telemetry.wrapper import telemetry
        assert not isinstance(telemetry, NNCFTelemetryStub)
    # cleanup
    importlib.reload(wrapper)


@pytest.mark.parametrize("env_var_to_define", [NNCF_CI_ENV_VAR_NAME,
                                               NNCF_DEV_ENV_VAR_NAME])
def test_telemetry_is_mocked_if_env_vars_defined(env_var_to_define, hide_pytest):
    with mock.patch.dict(os.environ, {env_var_to_define: "1"}, clear=True):
        # Need to reload the module where the logic concerning
        # env vars and telemetry object is evaluated
        from nncf.telemetry import wrapper
        importlib.reload(wrapper)

        # telemetry alias will no longer be available after reload,
        # so importing via a full name
        from nncf.telemetry.wrapper import telemetry
        assert isinstance(telemetry, NNCFTelemetryStub)
    # cleanup
    importlib.reload(wrapper)


@pytest.fixture(name="spies")
def spies_(request, mocker) -> Tuple[MagicMock, MagicMock, MagicMock]:
    from nncf.telemetry import telemetry
    send_event_spy = mocker.spy(telemetry, "send_event")
    start_session_event_spy = mocker.spy(telemetry, "start_session")
    end_session_event_spy = mocker.spy(telemetry, "end_session")
    return (send_event_spy, start_session_event_spy, end_session_event_spy)


CATEGORY_FOR_TEST = "test_category"
NAME_OF_EVENT_FOR_TEST = "test_category"
CONSTANT_DEFAULT_ARGVALUE = "foo"
EVENT_INT_DATA = 665


def test_tracked_function(mocker, spies):
    send_event_spy, start_session_event_spy, end_session_event_spy = spies

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
                      extractors=["arg2", NoArgExtractor("arg1"), "arg3", DoubleArgExtractor("arg2")])
    def fn_to_test(arg1, arg2, arg3 = CONSTANT_DEFAULT_ARGVALUE):
        pass

    fn_to_test("bar", "baz")
    assert start_session_event_spy.call_count == 1
    assert end_session_event_spy.call_count == 1
    assert send_event_spy.call_count == 4

    expected_call_args_list = [
        call(event_category=CATEGORY_FOR_TEST,
             event_action="arg2",
             event_label="baz",
             event_value=None),
        call(event_category=CATEGORY_FOR_TEST,
             event_action=NAME_OF_EVENT_FOR_TEST,
             event_label=None,
             event_value=None),
        call(event_category=CATEGORY_FOR_TEST,
             event_action="arg3",
             event_label=CONSTANT_DEFAULT_ARGVALUE,
             event_value=None),
        call(event_category=CATEGORY_FOR_TEST,
             event_action=NAME_OF_EVENT_FOR_TEST,
             event_label="bazbaz",
             event_value=EVENT_INT_DATA)
        ]

    assert send_event_spy.call_args_list == expected_call_args_list


CATEGORY_FOR_TEST2 = "test_category2"

@tracked_function(category=CATEGORY_FOR_TEST, extractors=["arg"])
def inner_same(arg):
    return arg

@tracked_function(category=CATEGORY_FOR_TEST2, extractors=["arg"])
def inner_other(arg):
    return arg

@tracked_function(category=CATEGORY_FOR_TEST, extractors=["arg"])
def outer(arg, same: bool):
    if same:
        return inner_same(arg)
    return inner_other(arg)

def test_nested_function_same_categories(mocker, spies):
    send_event_spy, start_session_event_spy, end_session_event_spy = spies

    outer("foo", same=True)

    assert start_session_event_spy.call_count == 1
    assert end_session_event_spy.call_count == 1
    assert send_event_spy.call_count == 2

    expected_call_args_list = [
        call(event_category=CATEGORY_FOR_TEST,
             event_action="arg",
             event_label="foo",
             event_value=None),
        call(event_category=CATEGORY_FOR_TEST,
             event_action="arg",
             event_label="foo",
             event_value=None) ]

    assert send_event_spy.call_args_list == expected_call_args_list

def test_nested_function_different_categories(mocker, spies):
    send_event_spy, start_session_event_spy, end_session_event_spy = spies

    outer("foo", same=False)

    assert start_session_event_spy.call_count == 2
    assert end_session_event_spy.call_count == 2
    assert send_event_spy.call_count == 2

    expected_call_args_list = [
        call(event_category=CATEGORY_FOR_TEST,
             event_action="arg",
             event_label="foo",
             event_value=None),
        call(event_category=CATEGORY_FOR_TEST2,
             event_action="arg",
             event_label="foo",
             event_value=None) ]

    assert send_event_spy.call_args_list == expected_call_args_list

    expected_session_call_args_list = [
        call(CATEGORY_FOR_TEST),
        call(CATEGORY_FOR_TEST2)
        ]


    assert start_session_event_spy.call_args_list == expected_session_call_args_list
    assert end_session_event_spy.call_args_list == list(reversed(expected_session_call_args_list))
