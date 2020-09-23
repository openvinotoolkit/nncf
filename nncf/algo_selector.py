"""
 Copyright (c) 2019-2020 Intel Corporation
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
# pylint:disable=relative-beyond-top-level
from .compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController, CompressionLevel
from .registry import Registry

COMPRESSION_ALGORITHMS = Registry('compression algorithm')


@COMPRESSION_ALGORITHMS.register('NoCompressionAlgorithmBuilder')
class NoCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    pass


# pylint:disable=abstract-method
class NoCompressionAlgorithmController(CompressionAlgorithmController):
    def compression_level(self) -> CompressionLevel:
        """
        Returns level of compression. Should be used on saving best checkpoints to distinguish between
        uncompressed, partially compressed and fully compressed models.
        """
        return CompressionLevel.NONE
