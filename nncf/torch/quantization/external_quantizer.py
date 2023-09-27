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

from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.quantization.debug_interface import QuantizationDebugInterface

EXTERNAL_QUANTIZERS_STORAGE_NAME = "external_quantizers"
EXTERNAL_QUANTIZERS_STORAGE_PREFIX = "_nncf." + EXTERNAL_QUANTIZERS_STORAGE_NAME


class ExternalQuantizerCallHook:
    """
    Cannot simply register the quantizer module as a callable hook, since we need to call
    a thread-local version of the quantizer module during base module execution.
    """

    def __init__(
        self,
        context: TracingContext,
        quantizer_storage_key: str,
        debug_interface: QuantizationDebugInterface = None,
    ):
        self.compressed_context = context
        self.quantizer_storage_key = quantizer_storage_key
        self.debug_interface = debug_interface

    def __call__(self, *args, **kwargs):
        if self.debug_interface is not None:
            self.debug_interface.register_activation_quantize_call(str(self.quantizer_storage_key))
        replica = self.compressed_context.base_module_thread_local_replica
        storage = getattr(replica.nncf, EXTERNAL_QUANTIZERS_STORAGE_NAME)
        return storage[self.quantizer_storage_key](*args, **kwargs)
