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

__version__ = "2.5.0"

BKC_TORCH_VERSION = "2.0.1"
BKC_TORCHVISION_VERSION = "0.15.1"
BKC_TF_VERSION = "2.12.*"

def get_version(version):
    from setuptools_scm.version import guess_next_version
    return version.format_next_version(guess_next_version, '{guessed}b{distance}')