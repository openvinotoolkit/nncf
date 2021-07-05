"""
 Copyright (c) 2021 Intel Corporation
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


class StatefulClassesRegistry:
    REQUIRED_METHOD_NAME = 'from_state'

    def __init__(self):
        self._stateful_classes = dict()
        self._stateful_class_names = dict()

    def register(self, name=None):
        def decorator(cls):
            class_name = name if name is not None else cls.__name__

            if class_name in self._stateful_class_names:
                raise ValueError(
                    '{} has already been registered to {}'.format(class_name, self._stateful_classes[class_name]))

            if cls in self._stateful_class_names:
                raise ValueError('{} has already been registered to {}'.format(cls, self._stateful_class_names[cls]))

            if inspect.isclass(cls) and not hasattr(cls, self.REQUIRED_METHOD_NAME):
                raise ValueError('Cannot register a class ({}) that does not have {}() method.'.format(
                    class_name, self.REQUIRED_METHOD_NAME))

            self._stateful_class_names[cls] = class_name
            self._stateful_classes[class_name] = cls

            return cls

        return decorator

    def get_registered_class(self, class_name):
        if class_name in self._stateful_classes:
            return self._stateful_classes[class_name]
        raise KeyError('No registered stateful classes with {} name'.format(class_name))

    def get_registered_class_name(self, stateful_cls):
        if stateful_cls in self._stateful_class_names:
            return self._stateful_class_names[stateful_cls]
        raise KeyError('No registered stateful class names for {} class'.format(stateful_cls.__name__))


class CommonStatefulClassesRegistry:
    @staticmethod
    def register(name=None):
        def decorator(cls):
            PT_STATEFUL_CLASSES.register(name)(cls)
            TF_STATEFUL_CLASSES.register(name)(cls)
            return cls

        return decorator

    @staticmethod
    def get_registered_class(class_name):
        return PT_STATEFUL_CLASSES.get_registered_class(class_name)

    @staticmethod
    def get_registered_class_name(stateful_cls):
        return PT_STATEFUL_CLASSES.get_registered_class_name(stateful_cls)


PT_STATEFUL_CLASSES = StatefulClassesRegistry()
TF_STATEFUL_CLASSES = StatefulClassesRegistry()
