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

import inspect
from typing import Callable, Dict, TypeVar

TObj = TypeVar("TObj", bound=type)


class StatefulClassesRegistry:
    """
    Registry for the stateful classes  - classes that can be restored from their state by `from_state` method.
    """

    REQUIRED_METHOD_NAME = "from_state"

    def __init__(self) -> None:
        self._name_vs_class_map: Dict[str, type] = {}
        self._class_vs_name_map: Dict[type, str] = {}

    def register(self, name: str = None) -> Callable[[TObj], TObj]:
        """
        Decorator to map class with some name - specified in the argument or name of the class.

        :param name: The registration name. By default, it's name of the class.
        :return: The inner function for registration.
        """

        def decorator(cls: TObj) -> TObj:
            registered_name = name if name is not None else cls.__name__

            if registered_name in self._name_vs_class_map:
                raise ValueError(
                    "{} has already been registered to {}".format(
                        registered_name, self._name_vs_class_map[registered_name]
                    )
                )

            if cls in self._class_vs_name_map:
                raise ValueError("{} has already been registered to {}".format(cls, self._class_vs_name_map[cls]))

            if inspect.isclass(cls) and not hasattr(cls, self.REQUIRED_METHOD_NAME):
                raise ValueError(
                    "Cannot register a class ({}) that does not have {}() method.".format(
                        registered_name, self.REQUIRED_METHOD_NAME
                    )
                )

            self._class_vs_name_map[cls] = registered_name
            self._name_vs_class_map[registered_name] = cls

            return cls

        return decorator

    def get_registered_class(self, registered_name: str) -> type:
        """
        Provides a class that was registered with the given name.

        :param registered_name: name
        :return: class that was registered with the given name
        """
        if registered_name in self._name_vs_class_map:
            return self._name_vs_class_map[registered_name]
        raise KeyError("No registered stateful classes with {} name".format(registered_name))

    def get_registered_name(self, stateful_cls: type) -> str:
        """
        Provides a name that was used to register the given stateful class.

        :param stateful_cls: class
        :return: name that was used on registration of the given class
        """
        if stateful_cls in self._class_vs_name_map:
            return self._class_vs_name_map[stateful_cls]
        raise KeyError("The class {} was not registered.".format(stateful_cls.__name__))


class CommonStatefulClassesRegistry:
    """
    Common for TF and PT registry for the stateful classes.
    """

    @staticmethod
    def register(name: str = None) -> Callable[[TObj], TObj]:
        """
        Decorator to map class with some name - specified in the argument or name of the class.

        :param name: The registration name. By default, it's name of the class.
        :return: The inner function for registration.
        """

        def decorator(cls: TObj) -> TObj:
            PT_STATEFUL_CLASSES.register(name)(cls)
            TF_STATEFUL_CLASSES.register(name)(cls)
            return cls

        return decorator

    @staticmethod
    def get_registered_class(registered_name: str) -> type:
        """
        Provides a class that was registered with the given name.

        :param registered_name: name
        :return: class that was registered with the given name
        """
        return PT_STATEFUL_CLASSES.get_registered_class(registered_name)

    @staticmethod
    def get_registered_name(stateful_cls: type) -> str:
        """
        Provides a name that was used to register the given stateful class.

        :param stateful_cls: class
        :return: name that was used on registration of the given class
        """
        return PT_STATEFUL_CLASSES.get_registered_name(stateful_cls)


PT_STATEFUL_CLASSES = StatefulClassesRegistry()
TF_STATEFUL_CLASSES = StatefulClassesRegistry()
