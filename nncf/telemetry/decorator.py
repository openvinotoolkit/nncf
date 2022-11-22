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

import inspect
from typing import Callable
from typing import List, Union
from nncf.telemetry.wrapper import NNCFTelemetry
from nncf.telemetry.extractors import TelemetryExtractor
from nncf.telemetry.extractors import VerbatimTelemetryExtractor
from nncf.telemetry.events import get_current_category
from nncf.telemetry.events import set_current_category
from nncf.telemetry.events import unset_current_category


class tracked_function:
    def __init__(self, category: str = None,
            collectors: List[Union[str, TelemetryExtractor]] = None):
        self._category = category
        self._collectors = [VerbatimTelemetryExtractor(x) if isinstance(x, str) else x for x in collectors]

    def __call__(self, fn: Callable) -> Callable:
        fn_signature = inspect.signature(fn)

        def wrapped(*args, **kwargs):
            bound_args = fn_signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            events = []  # type: CollectedEvent
            for collector in self._collectors:
                argname = collector.argname
                argvalue = bound_args.arguments[argname] if argname is not None else None
                event = collector.extract(argvalue)
                events.append(event)

            if self._category is not None:
                NNCFTelemetry.start_session(self._category)
                set_current_category(self._category)

            category = get_current_category()
            if category is not None:
                for event in events:
                    NNCFTelemetry.send_event(event_category=category,
                                             event_action=event.name,
                                             event_label=event.data,
                                             event_value=event.int_data)
            retval = fn(*args, **kwargs)

            if category is not None:
                NNCFTelemetry.end_session(self._category)
                unset_current_category()

            return retval

        return wrapped

