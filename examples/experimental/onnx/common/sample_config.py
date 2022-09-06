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
import argparse


class ActionWrapper(argparse.Action):
    def __init__(self, action):
        self._action = action
        super().__init__(action.option_strings, action.dest, nargs=action.nargs, const=action.const,
                         default=action.default, type=action.type, choices=action.choices, required=action.required,
                         help=action.help, metavar=action.metavar)
        self._action = action

    def __getattr__(self, item):
        return getattr(self._action, item)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.seen_actions.add(self._action.dest)
        return self._action(parser, namespace, values, option_string)


# pylint: disable=protected-access
class CustomArgumentGroup(argparse._ArgumentGroup):
    def _add_action(self, action):
        super()._add_action(ActionWrapper(action))


# pylint: disable=protected-access
class CustomActionContainer(argparse._ActionsContainer):
    def add_argument_group(self, *args, **kwargs):
        group = CustomArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group


class CustomArgumentParser(CustomActionContainer, argparse.ArgumentParser):
    """ArgumentParser that saves which arguments are provided"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.seen_actions = set()

    def parse_known_args(self, args=None, namespace=None):
        self.seen_actions.clear()
        return super().parse_known_args(args, namespace)
