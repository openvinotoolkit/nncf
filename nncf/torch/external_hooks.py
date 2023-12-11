# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.quantization.debug_interface import QuantizationDebugInterface

EXTERNAL_OP_STORAGE_NAME = "external_op"
EXTERNAL_QUANTIZERS_STORAGE_NAME = "external_quantizers"
EXTERNAL_QUANTIZERS_STORAGE_PREFIX = "_nncf." + EXTERNAL_QUANTIZERS_STORAGE_NAME


class ExternalOpCallHook:
    """
    Hook which is calling operation registered in the NNCFInterface
    by given storage name and storage key. Target operation should be
    registered before the ExternalOpCallHook call.
    Hook module could not be registered as a callable hook
    since a thread-local version of the module should be used during
    the base module execution.
    """

    def __init__(self, storage_name: str, context: TracingContext, storage_key: str):
        """
        :param storage_name: Attribute name of a model NNCFInterface.
        :param context: Current tracing context.
        :param storage_key: Key to retrieve callable hook
        """
        self._storage_name = storage_name
        self._compressed_context = context
        self._storage_key = storage_key

    def __call__(self, *args: Any, **kwargs) -> Any:
        replica = self._compressed_context.base_module_thread_local_replica
        storage = getattr(replica.nncf, self._storage_name)
        return storage[self._storage_key](*args, **kwargs)


class ExternalQuantizerCallHook(ExternalOpCallHook):
    """
    External hook which is using quantization storage name and
    could be constructed with a debug interface.
    """

    def __init__(
        self,
        context: TracingContext,
        quantizer_storage_key: str,
        debug_interface: QuantizationDebugInterface = None,
    ):
        super().__init__(EXTERNAL_QUANTIZERS_STORAGE_NAME, context, quantizer_storage_key)
        self.debug_interface = debug_interface

    def __call__(self, *args, **kwargs):
        if self.debug_interface is not None:
            self.debug_interface.register_activation_quantize_call(str(self._storage_key))
        return super().__call__(*args, **kwargs)
