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
import functools
import inspect
from typing import Any, Callable, List, Optional, TypeVar, Union

from nncf.telemetry.events import MODEL_BASED_CATEGORY
from nncf.telemetry.events import get_current_category
from nncf.telemetry.events import get_model_based_category
from nncf.telemetry.events import telemetry_category
from nncf.telemetry.extractors import CollectedEvent
from nncf.telemetry.extractors import TelemetryExtractor
from nncf.telemetry.extractors import VerbatimTelemetryExtractor
from nncf.telemetry.wrapper import telemetry

TFunction = TypeVar("TFunction", bound=Callable[..., Any])


class tracked_function:
    """
    Decorator to add telemetry events to a given function execution. The call to a decorated function
    will in general result in a telemetry session being created and a set of events being sent before
    function execution. The category of the session and events will be determined by parameters to the decorator.
    """

    def __init__(self, category: str = None, extractors: Optional[List[Union[str, TelemetryExtractor]]] = None) -> None:
        """
        :param category: A category to be attributed to the events. If set to None, no events will be sent.
        :param extractors: Add argument names in this list as string values to send an event with an "action" equal to
            the argument name and "label" equal to the argument value before the function start.
            The argument must be either a string or a dictionary of strings. If that is not the case, instead of
            argument names you can specify an object of a customized `TelemetryExtractor` subclass; use the same
            approach for more complex event reporting.
        """
        self._category = category
        if extractors is not None:
            self._collectors = [VerbatimTelemetryExtractor(x) if isinstance(x, str) else x for x in extractors]
        else:
            self._collectors = []

    def __call__(self, fn: TFunction) -> TFunction:
        fn_signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound_args = fn_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if self._category == MODEL_BASED_CATEGORY:
                model_argument = bound_args.arguments.get("model", None)
                self._category = get_model_based_category(model_argument)

            events: List[CollectedEvent] = []
            for collector in self._collectors:
                argname = collector.argname
                argvalue = bound_args.arguments[argname] if argname else None
                event = collector.extract(argvalue)
                events.append(event)

            previous_category = get_current_category()
            with telemetry_category(self._category) as category:
                if category is not None:
                    if category != previous_category:
                        telemetry.start_session(self._category)
                    for event in events:
                        telemetry.send_event(
                            event_category=category,
                            event_action=event.name,
                            event_label=event.data,
                            event_value=event.int_data,
                        )

                retval = fn(*args, **kwargs)

                if category is not None and category != previous_category:
                    telemetry.end_session(self._category)
            return retval

        return wrapped  # type: ignore[return-value]
