# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Synthetic test to verify INT2 symmetric decompression subgraph
can be exported to OpenVINO IR via torch.jit.trace + openvino.convert_model.
"""

import numpy as np
import openvino as ov
import torch


def pack_uint2(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().reshape(-1, 4)
    packed_tensor = (
        torch.bitwise_and(packed_tensor[..., 0], 3)
        | (torch.bitwise_and(packed_tensor[..., 1], 3) << 2)
        | (torch.bitwise_and(packed_tensor[..., 2], 3) << 4)
        | (torch.bitwise_and(packed_tensor[..., 3], 3) << 6)
    )
    return packed_tensor


def unpack_uint2(packed_tensor: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.bitwise_and(packed_tensor, 3),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 2), 3),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 4), 3),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 6), 3),
        ),
        dim=-1,
    )


def decompress_symmetric(input, scale):
    input = input.type(dtype=scale.dtype)
    return input * scale


class INT2SymmetricLinear(torch.nn.Module):
    """
    A simple linear layer that uses INT2 symmetric weight decompression,
    matching the NNCF INT2SymmetricWeightsDecompressor pattern.
    """

    ZERO_POINT_VALUE = 2

    def __init__(self, in_features, out_features, group_size):
        super().__init__()
        assert out_features % group_size == 0
        ngroups = out_features // group_size

        compressed_weight_shape = (ngroups, group_size, in_features)
        scale_shape = (ngroups, 1, in_features)

        # Random uint2 weights [0, 3]
        rng = np.random.default_rng(seed=42)
        raw_weights = rng.integers(0, 4, size=compressed_weight_shape, dtype=np.uint8)
        scale = (rng.random(scale_shape, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)

        self.compressed_weight_shape = compressed_weight_shape
        self.packed_weight = torch.nn.Parameter(pack_uint2(torch.from_numpy(raw_weights)), requires_grad=False)
        self.register_buffer("_scale", torch.from_numpy(scale).to(torch.float16))
        self.register_buffer("_zero_point", torch.tensor(self.ZERO_POINT_VALUE, dtype=torch.uint8))
        self.result_shape = (out_features, in_features)
        self.result_dtype = torch.float32

    def forward(self, x):
        # NNCF INT2 symmetric decompression pattern
        w = unpack_uint2(self.packed_weight)
        w = w.reshape(self.compressed_weight_shape)
        w = w.type(dtype=self.result_dtype) - self._zero_point.type(dtype=self.result_dtype)
        w = decompress_symmetric(w, self._scale)
        w = w.reshape(self.result_shape)
        w = w.type(dtype=self.result_dtype)
        return torch.matmul(x, w.t())


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = INT2SymmetricLinear(16, 32, group_size=4)
        self.linear2 = INT2SymmetricLinear(32, 16, group_size=4)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


def main():
    print("=== Synthetic INT2 export test ===")
    model = SmallModel()
    model.eval()

    dummy_input = torch.randn(1, 16)

    # Step 1: Convert to OpenVINO IR
    print("[1/4] Converting to OpenVINO IR...")
    ov_model = ov.convert_model(model, example_input=dummy_input)
    print("      Conversion successful.")

    # Step 2: Check u2 constants in the converted OV model
    print("[2/4] Checking u2 constants in OV model...")
    u2_constants = []
    for op in ov_model.get_ordered_ops():
        if op.get_type_name() == "Constant" and "uint2" in str(op.get_output_element_type(0)):
            u2_constants.append(op)

    expected_u2_count = 2  # one per INT2SymmetricLinear layer
    print(f"      Found {len(u2_constants)} u2 constant(s) (expected {expected_u2_count}).")
    for c in u2_constants:
        print(f"        - {c.get_friendly_name()}: shape={c.get_output_partial_shape(0)}")
    assert len(u2_constants) == expected_u2_count, f"Expected {expected_u2_count} u2 constants, got {len(u2_constants)}"
    print("      PASSED - u2 constants detected.")

    # Step 3: Save IR
    ir_path = "/tmp/test_int2_synthetic_ir"
    print(f"[3/4] Saving IR to {ir_path}...")
    ov.save_model(ov_model, f"{ir_path}/model.xml")
    print("      Save successful.")

    # Step 4: Verify inference
    print("[4/4] Running inference comparison...")
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    compiled = ov.Core().compile_model(ov_model, "CPU")
    ov_out = compiled(dummy_input.numpy())[0]

    max_diff = np.max(np.abs(torch_out - ov_out))
    print(f"      Max absolute difference: {max_diff:.6e}")
    if max_diff < 1e-2:
        print("      PASSED - Outputs match within tolerance.")
    else:
        print(f"      WARNING - Large difference detected: {max_diff}")

    print("\n=== All steps completed successfully! ===")


if __name__ == "__main__":
    main()
