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


from dataclasses import dataclass
from typing import List, Optional

import pytest

from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.experimental.common.pruning.operations import ReshapeMode
from nncf.experimental.common.pruning.operations import ReshapePruningOp


@dataclass
class ReshapeParsingDesc:
    input_shape: List[int]
    output_shape: List[int]
    mode: ReshapeMode = ReshapeMode.DEFAULT
    in_map: Optional[ReshapePruningOp.DIMENSION_MAP] = None
    out_map: Optional[ReshapePruningOp.DIMENSION_MAP] = None

    def __str__(self):
        return "_".join(map(str, self.input_shape)) + " -> " + "_".join(map(str, self.output_shape))


RESHAPE_PARSING_DESCS = [
    ReshapeParsingDesc(
        input_shape=[2, 3, 2, 2, 5],
        output_shape=[6, 4, 5],
        mode=ReshapeMode.SHRINK,
        in_map={0: [0], 1: [0], 2: [1], 3: [1], 4: [2]},
        out_map={0: [0, 1], 1: [2, 3], 2: [4]},
    ),
    ReshapeParsingDesc(
        input_shape=[6, 4, 5],
        output_shape=[2, 3, 2, 2, 5],
        mode=ReshapeMode.EXTEND,
        in_map={0: [0, 1], 1: [2, 3], 2: [4]},
        out_map={0: [0], 1: [0], 2: [1], 3: [1], 4: [2]},
    ),
    ReshapeParsingDesc(
        input_shape=[6, 4, 5],
        output_shape=[1, 2, 3, 2, 1, 2, 5],
        mode=ReshapeMode.EXTEND,
        in_map={0: [1, 2], 1: [3, 5], 2: [6]},
        out_map={1: [0], 2: [0], 3: [1], 5: [1], 6: [2]},
    ),
    ReshapeParsingDesc(
        input_shape=[2], output_shape=[1, 2], mode=ReshapeMode.IDENTITY_WITHOUT_ONES, in_map={0: [1]}, out_map={1: [0]}
    ),
    ReshapeParsingDesc(
        input_shape=[1, 2],
        output_shape=[2],
        mode=ReshapeMode.IDENTITY_WITHOUT_ONES,
        in_map={1: [0]},
        out_map={0: [1]},
    ),
    ReshapeParsingDesc(
        input_shape=[2, 1],
        output_shape=[2],
        mode=ReshapeMode.IDENTITY_WITHOUT_ONES,
        in_map={0: [0]},
        out_map={0: [0]},
    ),
    ReshapeParsingDesc(
        input_shape=[2],
        output_shape=[2, 1],
        mode=ReshapeMode.IDENTITY_WITHOUT_ONES,
        in_map={0: [0]},
        out_map={0: [0]},
    ),
    ReshapeParsingDesc(
        input_shape=[6, 4, 5],
        output_shape=[2, 3, 20],
    ),
    ReshapeParsingDesc(
        input_shape=[4, 2, 3],
        output_shape=[2, 3, 4],
    ),
    ReshapeParsingDesc(
        input_shape=[3, 2, 1, 1],
        output_shape=[6],
        mode=ReshapeMode.SHRINK,
        in_map={0: [0], 1: [0]},
        out_map={0: [0, 1]},
    ),
    ReshapeParsingDesc(
        input_shape=[6],
        output_shape=[3, 2, 1, 1],
        mode=ReshapeMode.EXTEND,
        in_map={0: [0, 1]},
        out_map={0: [0], 1: [0]},
    ),
    ReshapeParsingDesc(
        input_shape=[1, 1, 2, 1, 8, 1],
        output_shape=[2, 1, 1, 1, 1, 8],
        mode=ReshapeMode.IDENTITY_WITHOUT_ONES,
        in_map={2: [0], 4: [5]},
        out_map={0: [2], 5: [4]},
    ),
]


@pytest.mark.parametrize("desc", RESHAPE_PARSING_DESCS, ids=map(str, RESHAPE_PARSING_DESCS))
def test_reshape_parsing(desc: ReshapeParsingDesc):
    attrs = ReshapeLayerAttributes(desc.input_shape, desc.output_shape)

    in_map, out_map, mode = ReshapePruningOp.parse_reshape(attrs)

    assert mode == desc.mode
    assert in_map == desc.in_map
    assert out_map == desc.out_map
