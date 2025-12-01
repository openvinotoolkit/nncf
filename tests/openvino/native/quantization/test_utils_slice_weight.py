import numpy as np
import pytest
import torch
from nncf.quantization.algorithms.weight_compression import utils


@pytest.mark.parametrize(
    "shape, transpose_b, start, end",
    [
        # transpose_b=True means weight layout is [out_features, in_features] -> slice columns
        ((5, 8), True, 1, 4),
        ((3, 6), True, 0, 3),
        # transpose_b=False means weight layout is [in_features, out_features] -> slice rows
        ((8, 5), False, 2, 6),
        ((6, 3), False, 0, 2),
    ],
)
def test_slice_and_assign_weight_block(shape, transpose_b, start, end):
    """
    Verify slice_weight returns the expected sub-block and assign_weight_slice writes it back
    in the correct orientation for both transpose_b True and False.
    """

    weight = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    block = utils.slice_weight(weight, start, end, transpose_b)

    # Expected block depending on transpose_b semantics
    if transpose_b:
        expected_block = weight[:, start:end]
    else:
        expected_block = weight[start:end, :]

    # The returned block should match the expected slice
    np.testing.assert_array_equal(block, expected_block)

    # Prepare a new block to assign (different values)
    new_block = np.full(expected_block.shape, fill_value=123, dtype=weight.dtype)

    # Assign it back using the helper
    utils.assign_weight_slice(weight, start, end, new_block, transpose_b)
    if transpose_b:
        np.testing.assert_array_equal(weight[:, start:end], new_block)
    else:
        np.testing.assert_array_equal(weight[start:end, :], new_block)

def test_zero_mask_columns():
    """
    Verifies that zero_mask_columns correctly zeros out channels 
    based on the boolean mask and transpose_b setting.
    """
    shape = (4, 4)
    # Create a mask: e.g., index 1 and 3 are True (should be zeroed)
    mask = np.array([False, True, False, True]) 
    
    # CASE 1: transpose_b=True (Layout [Out, In] -> Columns are inputs)
    weight = np.ones(shape, dtype=np.int32)
    utils.zero_mask_columns(weight, mask, transpose_b=True)
    
    # Columns 1 and 3 should be 0, others 1
    expected = np.ones(shape, dtype=np.int32)
    expected[:, mask] = 0 
    np.testing.assert_array_equal(weight, expected)

    # CASE 2: transpose_b=False (Layout [In, Out] -> Rows are inputs)
    weight = np.ones(shape, dtype=np.int32)
    utils.zero_mask_columns(weight, mask, transpose_b=False)
    
    # Rows 1 and 3 should be 0, others 1
    expected = np.ones(shape, dtype=np.int32)
    expected[mask, :] = 0
    np.testing.assert_array_equal(weight, expected)




def test_slice_utils_pytorch_compatibility():
    """
    Ensures the helpers work with torch.Tensor objects, not just numpy arrays.
    """
    # [In, Out] = [4, 2]
    # transpose_b=False
    weight = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    # 1. Test Slicing (taking middle 2 rows)
    block = utils.slice_weight(weight, 1, 3, transpose_b=False)
    assert torch.equal(block, torch.tensor([[3, 4], [5, 6]]))
    
    # 2. Test Assigning
    new_data = torch.tensor([[10, 10], [10, 10]])
    utils.assign_weight_slice(weight, 1, 3, new_data, transpose_b=False)
    
    expected = torch.tensor([[1, 2], [10, 10], [10, 10], [7, 8]])
    assert torch.equal(weight, expected)
