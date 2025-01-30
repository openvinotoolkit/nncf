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


import builtins
import collections
import importlib
import pickle  # nosec

# Regular unpickling is prone to arbitrary code execution attacks.
# This module implements the objects required by torch.load from a
# pickler module to be able to work, but the Unpickler object provided
# here is restricted with respect to the classes it can load, to safeguard
# against attacks described above.
# ** WARNING **: there is no final guarantee that using this module in
# conjunction with torch.load mitigates all possible routes of arbitrary code
# execution attacks possible while using Python's `pickle`.
# ** Only load the data you trust **

load = pickle.load


class Unpickler(pickle.Unpickler):
    safe_builtins = {"range", "complex", "set", "frozenset", "slice", "dict"}
    safe_collections = {"OrderedDict", "defaultdict"}

    allowed_classes = {
        "torch": {"Tensor", "FloatStorage", "LongStorage", "IntStorage"},
        "torch._utils": {"_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_parameter"},
        "torch.nn": {"Module"},
        "torch.optim.adam": {"Adam"},
        "nncf.api.compression": {"CompressionStage", "CompressionLevel"},
        "nncf.common.quantization.structs": {"QuantizationScheme"},
        "numpy.core.multiarray": {"scalar"},  # numpy<2
        "numpy._core.multiarray": {"scalar"},  # numpy>=2
        "numpy": {"dtype"},
        "_codecs": {"encode"},
    }

    def find_class(self, module_name, class_name):
        # Only allow safe classes from builtins.
        if module_name in ["builtins", "__builtin__"] and class_name in Unpickler.safe_builtins:
            return getattr(builtins, class_name)
        if module_name == "collections" and class_name in Unpickler.safe_collections:
            return getattr(collections, class_name)
        for allowed_module_name, val in Unpickler.allowed_classes.items():
            if module_name == allowed_module_name and class_name in val:
                module = importlib.import_module(module_name)
                return getattr(module, class_name)

        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module_name, class_name))
