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


from nncf.torch.nncf_network import NNCFNetwork


def strip(model: NNCFNetwork, do_copy: bool = True) -> NNCFNetwork:
    """
    Returns the model object with as much custom NNCF additions as possible removed
    while still preserving the functioning of the model object as a compressed model.

    :param do_copy: If True (default), will return a copy of the currently associated model object. If False,
      will return the currently associated model object "stripped" in-place.
    :return: The stripped model.
    """
    return model.nncf.strip(do_copy)
