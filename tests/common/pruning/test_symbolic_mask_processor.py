import pytest

from nncf.common.pruning.symbolic_mask import AmbiguousSymbolicMask
from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.symbolic_mask import SymbolicMaskProcessor
from nncf.common.pruning.symbolic_mask import SymbolicMaskProducer


@pytest.mark.parametrize("shape,raise_runtime_error", [(5, False), ([6], False), ([1, 2], True)])
def test_ones(shape, raise_runtime_error):
    device = None
    if raise_runtime_error:
        with pytest.raises(RuntimeError):
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
        assert not mask_producers[idx] is repeated_producer
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
