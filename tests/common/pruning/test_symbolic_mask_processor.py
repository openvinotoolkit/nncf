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

import pytest

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.pruning.symbolic_mask import AmbiguousSymbolicMask
from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.symbolic_mask import SymbolicMaskProcessor
from nncf.common.pruning.symbolic_mask import SymbolicMaskProducer
from nncf.experimental.common.pruning.operations import ElementwisePruningOp
from nncf.experimental.common.pruning.propagation_data import PropagationGroup
from nncf.experimental.common.pruning.propagation_data import PropagationMask
from nncf.experimental.common.pruning.propagation_data import PruningBlock


@pytest.mark.parametrize("shape,raise_runtime_error", [(5, False), ([6], False), ([1, 2], True)])
def test_ones(shape, raise_runtime_error):
    device = None
    if raise_runtime_error:
        with pytest.raises(nncf.ValidationError):
            tensor = SymbolicMaskProcessor.ones(shape, device)
    else:
        tensor = SymbolicMaskProcessor.ones(shape, device)
        assert tensor.mask_producers == []
        assert len(tensor.shape) == 1
        assert tensor.shape[0] == shape[0] if isinstance(shape, list) else shape
        assert tensor.device is None


def test_repeat():
    repeats = 5
    mask_producers = [SymbolicMaskProducer(3), SymbolicMaskProducer(5, 8)]
    mask = SymbolicMask(10, mask_producers)
    repeated_tensor = SymbolicMaskProcessor.repeat(mask, repeats)
    for idx, repeated_producer in enumerate(repeated_tensor.mask_producers):
        assert mask_producers[idx] is not repeated_producer
        assert mask_producers[idx].id == repeated_producer.id
        assert mask_producers[idx].sparse_multiplier * repeats == repeated_producer.sparse_multiplier


@pytest.mark.parametrize("consistent", [True, False])
def test_concat_inconsistent_sparse_multiplier(consistent):
    mask0 = SymbolicMask(8, [SymbolicMaskProducer(1, 2)])
    mask1 = SymbolicMask(4, [SymbolicMaskProducer(1, 2 if consistent else 4)])
    masks = [mask0, mask1]
    if not consistent:
        with pytest.raises(AssertionError):
            SymbolicMaskProcessor.concatenate(masks, axis=0)
        return

    concated_mask = SymbolicMaskProcessor.concatenate(masks, axis=0)
    assert concated_mask.shape[0] == 12
    assert len(concated_mask.mask_producers) == 1
    assert concated_mask.mask_producers[0].id == 1
    assert concated_mask.mask_producers[0].sparse_multiplier == 2


@pytest.mark.parametrize("masks_num", [1, 3])
def test_concat(masks_num):
    masks_producers = []
    for j in range(0, masks_num * 2, 2):
        masks_producers.append([SymbolicMaskProducer(i, i + 1) for i in range(j, j + 2)])

    masks = [SymbolicMask(i, producers) for i, producers in enumerate(masks_producers)]
    concated_mask = SymbolicMaskProcessor.concatenate(masks, axis=0)
    assert concated_mask.shape[0] == sum([mask.shape[0] for mask in masks])
    assert len(concated_mask.mask_producers) == len(masks) * 2
    for idx, mask_producer in enumerate(concated_mask.mask_producers):
        cur_mask_producer = masks_producers[idx // 2][idx % 2]
        assert cur_mask_producer is mask_producer


def test_empty_concat():
    empty_concat = SymbolicMaskProcessor.concatenate([], axis=0)
    assert empty_concat.shape[0] == 0
    assert not empty_concat.mask_producers


def test_concat_no_producers():
    concated_masks = SymbolicMaskProcessor.concatenate([SymbolicMask(2), SymbolicMask(3)], axis=0)
    assert concated_masks.shape[0] == 5
    assert not concated_masks.mask_producers


@pytest.mark.parametrize("all_close", [False, True])
def test_assert_all_close(all_close):
    tensors = [SymbolicMask(5 if all_close else i) for i in range(3)]
    if not all_close:
        with pytest.raises(AssertionError):
            SymbolicMaskProcessor.assert_allclose(tensors)
    else:
        SymbolicMaskProcessor.assert_allclose(tensors)


@pytest.mark.parametrize("all_close", [False, True])
def test_elementwise_mask_propagation(all_close):
    masks_producers = [[SymbolicMaskProducer(i)] for i in range(3)]
    masks = [SymbolicMask(5 if all_close else i, producer) for i, producer in enumerate(masks_producers)]
    result = SymbolicMaskProcessor.elementwise_mask_propagation(masks)
    assert {p.id for p in result.mask_producers} == set(range(3))
    if not all_close:
        assert isinstance(result, AmbiguousSymbolicMask)
    else:
        assert result.shape[0] == 5


@pytest.mark.parametrize("consistent", [True, False])
def test_elementwise_mask_propagation_inconsistent_(consistent):
    mask0 = SymbolicMask(5, [SymbolicMaskProducer(4, 2)])
    mask1 = SymbolicMask(5, [SymbolicMaskProducer(4, 2 if consistent else 4)])
    masks = [mask0, mask1]
    if not consistent:
        with pytest.raises(AssertionError):
            SymbolicMaskProcessor.elementwise_mask_propagation(masks)
        return

    result = SymbolicMaskProcessor.elementwise_mask_propagation(masks)
    assert result.shape[0] == 5
    assert len(result.mask_producers) == 1
    assert result.mask_producers[0].id == 4
    assert result.mask_producers[0].sparse_multiplier == 2


def idfn(val):
    if isinstance(val, tuple):
        return "_".join(str(x) for x in val)


@pytest.mark.parametrize(
    "shape_1, shape_2, ref_dim",
    (
        ((10,), (1, 2, 10, 10), 1),
        ((10, 10), (1, 2, 10, 10), 1),
        ((1, 1, 1, 10), (1, 2, 10, 10), 1),
        ((1, 1, 10, 10), (1, 2, 10, 10), 1),
        ((1, 2, 10, 10), (1, 2, 10, 10), None),
        ((1, 10), (10, 1), None),
        ((1, 1, 1, 1, 10), (1, 2, 10, 10), 2),
        ((1, 2, 1, 1, 10), (1, 2, 10, 10), None),
        ((1, 1, 2, 1, 10), (1, 2, 10, 10), None),
    ),
    ids=idfn,
)
@pytest.mark.parametrize("in_port_1, in_port_2", ((1, 0), (0, 1)), ids=("direct", "revers"))
def test_elementwise_mask_propagation_with_one_none_mask(shape_1, shape_2, ref_dim, in_port_1, in_port_2):
    graph = NNCFGraph()
    node_1 = graph.add_nncf_node("node_1", "", node_metatype=None)
    node_2 = graph.add_nncf_node("node_2", "", node_metatype=None)
    node_3 = graph.add_nncf_node("node_3", "", node_metatype=None)
    graph.add_edge_between_nncf_nodes(node_1.node_id, node_3.node_id, shape_1, in_port_1, 0, float)
    graph.add_edge_between_nncf_nodes(node_2.node_id, node_3.node_id, shape_2, in_port_2, 0, float)
    node_1.attributes["output_mask"] = None
    node_2.attributes["output_mask"] = PropagationMask({1: [PropagationGroup(PruningBlock())]})

    ElementwisePruningOp.mask_propagation(node_3, graph, None)

    if ref_dim is None:
        assert node_3.attributes["output_mask"] is None
    else:
        assert ref_dim in node_3.attributes["output_mask"].dim_groups_map


@pytest.mark.parametrize("consistent,out_shapes", [(True, [2, 2, 3]), (False, [2, 2, 4]), (False, [2, 2, 3, 0])])
def test_split(consistent, out_shapes):
    mask_producers = [SymbolicMaskProducer(1), SymbolicMaskProducer(2)]
    input_mask = SymbolicMask(7, mask_producers)
    if not consistent:
        with pytest.raises(AssertionError):
            SymbolicMaskProcessor.split(input_mask, out_shapes)
        return

    splitted_masks = SymbolicMaskProcessor.split(input_mask, out_shapes)
    assert len(splitted_masks) == 3
    for mask, mask_shape_ref in zip(splitted_masks, out_shapes):
        assert mask.shape[0] == mask_shape_ref
        assert mask.mask_producers == mask_producers
