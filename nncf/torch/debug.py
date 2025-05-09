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

from torch.nn import Module

from nncf.common.logging import nncf_logger


class CallCountTracker:
    def __init__(self, name):
        self.name = name
        self.call_counts = {}

    def init_with_key_list(self, key_list: list):
        self.call_counts = {key: 0 for key in key_list}
        nncf_logger.debug(f"{self.name} tracker: registered {len(self.call_counts)} entries")

    def register_call(self, key, counts=None):
        if key not in self.call_counts:
            nncf_logger.debug(f"DEBUG: {self.name} tracker: called an unregistered module: {key}")
            return
        if counts is None:
            self.call_counts[key] += 1
        else:
            self.call_counts[key] = counts

    def get_never_called_keys(self) -> list[str]:
        return [k for k, v in self.call_counts.items() if v == 0]

    def get_overcalled_keys_with_call_counts(self) -> dict[str, int]:
        return {k: v for k, v in self.call_counts.items() if v > 1}

    def get_total_call_count(self) -> int:
        if self.call_counts:
            return sum(self.call_counts.values())
        return 0

    def reset(self):
        for key in self.call_counts:
            self.call_counts[key] = 0


class DebugInterface:
    def pre_forward_actions(self, module: Module):
        raise NotImplementedError

    def post_forward_actions(self, module: Module):
        raise NotImplementedError

    def init_actual(self, owner_model):
        raise NotImplementedError


def debuggable_forward(forward_func):
    from nncf.torch.nncf_network import NNCFNetwork

    @functools.wraps(forward_func)
    def decorated(self: NNCFNetwork, *args, **kwargs):
        if hasattr(self, "nncf") and self.nncf.debug_interface is not None:
            self.nncf.debug_interface.pre_forward_actions(module=self)
        retval = forward_func(self, *args, **kwargs)
        if hasattr(self, "nncf") and self.nncf.debug_interface is not None:
            self.nncf.debug_interface.post_forward_actions(module=self)
        return retval

    return decorated


class CombinedDebugInterface(DebugInterface):
    def __init__(self):
        self._interfaces: list[DebugInterface] = []

    def add_interface(self, interface: "DebugInterface"):
        self._interfaces.append(interface)

    def init_actual(self, owner_model: "NNCFNetwork"):  # noqa: F821
        for interface in self._interfaces:
            interface.init_actual(owner_model)

    def pre_forward_actions(self, module: Module):
        for interface in self._interfaces:
            interface.pre_forward_actions(module)

    def post_forward_actions(self, module: Module):
        for interface in self._interfaces:
            interface.post_forward_actions(module)
