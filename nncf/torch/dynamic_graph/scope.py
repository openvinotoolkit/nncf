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
import re
from copy import deepcopy
from typing import List

import nncf


class ScopeElement:
    def __init__(self, calling_module_class_name: str, calling_field_name: str = None):
        self.calling_module_class_name = calling_module_class_name
        self.calling_field_name = calling_field_name

    def __str__(self):
        if self.calling_field_name is None:
            return self.calling_module_class_name
        return "{cls}[{name}]".format(cls=self.calling_module_class_name, name=self.calling_field_name)

    def __eq__(self, other: "ScopeElement"):
        return (self.calling_module_class_name == other.calling_module_class_name) and (
            self.calling_field_name == other.calling_field_name
        )

    def __hash__(self):
        return hash((self.calling_module_class_name, self.calling_field_name))

    @staticmethod
    def from_str(string: str):
        matches = re.search(r"(.*)\[(.*)\]|(.*)", string)
        if matches is None:
            raise nncf.InternalError("Invalid scope element string")
        if matches.groups()[0] is None and matches.groups()[1] is None:
            return ScopeElement(matches.groups()[2])
        if matches.groups()[0] is not None and matches.groups()[1] is not None:
            return ScopeElement(matches.groups()[0], matches.groups()[1])
        raise nncf.InternalError("Could not parse the scope element string")


class Scope:
    def __init__(self, scope_elements: List[ScopeElement] = None):
        if scope_elements is not None:
            self.scope_elements = scope_elements
        else:
            self.scope_elements = []

    def __str__(self):
        return "/".join([str(scope_el) for scope_el in self.scope_elements])

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: "Scope"):
        return self.scope_elements == other.scope_elements

    def __getitem__(self, key) -> ScopeElement:
        return self.scope_elements[key]

    def __contains__(self, item: "Scope"):
        """Idiom: ('A/B/C' in 'A/B') == True"""
        if len(self.scope_elements) > len(item.scope_elements):
            return False
        for i, element in enumerate(self.scope_elements):
            if element != item.scope_elements[i]:
                return False
        return True

    def __add__(self, rhs: "Scope") -> "Scope":
        init_list = self.scope_elements + rhs.scope_elements
        return Scope(init_list)

    def copy(self) -> "Scope":
        return Scope(deepcopy(self.scope_elements))

    def push(self, scope_element: ScopeElement):
        self.scope_elements.append(scope_element)

    def pop(self) -> ScopeElement:
        return self.scope_elements.pop()

    @staticmethod
    def from_str(string: str) -> "Scope":
        if string:
            elts = string.split("/")
        else:
            elts = []
        return Scope([ScopeElement.from_str(s) for s in elts])

    def get_iteration_scopes(self) -> List[str]:
        results = []
        from nncf.torch.layers import ITERATION_MODULES

        for scope_element in self.scope_elements:
            if scope_element.calling_module_class_name in ITERATION_MODULES.registry_dict:
                results.append(scope_element.calling_module_class_name)
        return results
