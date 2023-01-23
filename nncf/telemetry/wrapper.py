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
import sys
from abc import ABC
from abc import abstractmethod
from typing import Callable
from unittest.mock import MagicMock

from nncf import __version__
from nncf.common.logging import nncf_logger
from nncf.definitions import NNCF_CI_ENV_VAR_NAME
from nncf.definitions import NNCF_DEV_ENV_VAR_NAME


NNCFTelemetryStub = MagicMock

class ITelemetry(ABC):
    # For recommendations on proper usage of categories, actions, labels and values, see:
    # https://support.google.com/analytics/answer/1033068

    @abstractmethod
    def start_session(self, category: str, **kwargs):
        """
        Sends a message about starting of a new session.

        :param kwargs: additional parameters
        :param category: the application code
        :return: None
        """
        pass

    @abstractmethod
    def send_event(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                   force_send=False, **kwargs):
        """
        Send single event.

        :param event_category: category of the event
        :param event_action: action of the event
        :param event_label: the label associated with the action
        :param event_value: the integer value corresponding to this label
        :param force_send: forces to send event ignoring the consent value
        :param kwargs: additional parameters
        :return: None
        """
        pass

    @abstractmethod
    def end_session(self, category: str, **kwargs):
        """
        Sends a message about ending of the current session.

        :param kwargs: additional parameters
        :param category: the application code
        :return: None
        """
        pass


def skip_if_raised(func: Callable[..., None]) -> Callable[..., None]:
    """
    For the calls on the decorated function the execution will continue from the statement after the function
    even if an exception was triggered inside the function. The function to be wrapped must return nothing or None.
    """
    def wrapped(*args, **kwargs):
        try:
            func()
        except Exception as e:
            nncf_logger.debug(f"Skipped calling {func.__name__} - internally triggered exception {e}")
    return wrapped


class NNCFTelemetry(ITelemetry):
    MEASUREMENT_ID = 'UA-17808594-29'

    def __init__(self):
        try:
            self._impl = Telemetry(app_name='nncf', app_version=__version__, tid=self.MEASUREMENT_ID)
        except Exception as e:
            nncf_logger.debug(f"Failed to instantiate telemetry object: exception {e}")

    @skip_if_raised
    def start_session(self, category: str, **kwargs):
        self._impl.start_session(category, **kwargs)

    @skip_if_raised
    def send_event(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                   force_send=False, **kwargs):
        self._impl.send_event(event_category, event_action, event_label, event_value, force_send, **kwargs)

    @skip_if_raised
    def end_session(self, category: str, **kwargs):
        self._impl.end_session(category, **kwargs)


try:
    from openvino_telemetry import Telemetry
    telemetry = NNCFTelemetry()
except ImportError:
    nncf_logger.debug("openvino_telemetry package not found. No telemetry will be sent.")
    telemetry = NNCFTelemetryStub()

# Currently the easiest way to disable telemetry in tests. Will break telemetry if pytest is used in NNCF package code.
_IS_IN_PYTEST_CONTEXT = "pytest" in sys.modules

if os.getenv(NNCF_CI_ENV_VAR_NAME) or os.getenv(NNCF_DEV_ENV_VAR_NAME) \
        or _IS_IN_PYTEST_CONTEXT:
    telemetry = NNCFTelemetryStub()
