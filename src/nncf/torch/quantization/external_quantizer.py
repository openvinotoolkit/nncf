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

from nncf.torch.external_hook import ExternalOpCallHook
from nncf.torch.quantization.debug_interface import QuantizationDebugInterface

EXTERNAL_QUANTIZERS_STORAGE_NAME = "external_quantizers"
EXTERNAL_QUANTIZERS_STORAGE_PREFIX = "_nncf." + EXTERNAL_QUANTIZERS_STORAGE_NAME


class ExternalQuantizerCallHook(ExternalOpCallHook):
    """
    External hook which is using quantization storage name and
    could be constructed with a debug interface.
    """

    def __init__(
        self,
        quantizer_storage_key: str,
        debug_interface: QuantizationDebugInterface = None,
    ):
        super().__init__(EXTERNAL_QUANTIZERS_STORAGE_NAME, quantizer_storage_key)
        self.debug_interface = debug_interface

    def __call__(self, *args, **kwargs):
        if self.debug_interface is not None:
            self.debug_interface.register_activation_quantize_call(str(self._storage_key))
        return super().__call__(*args, **kwargs)
