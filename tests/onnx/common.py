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

from typing import Optional

import numpy as np
import onnx

ONNX_NNCF_SEED = 42


def get_random_generator(seed: int = ONNX_NNCF_SEED):
    rng = np.random.default_rng(seed)
    return rng


class ModelBuilder:
    """
    Helper class for creating ONNX models for tests.
    """

    def __init__(self):
        self._nodes = []
        self._initializers = []
        self._inputs = []
        self._outputs = []
        self._graph_name = "onnx-graph"

    def add_input(self, name: str, shape: tuple[int]) -> str:
        self._inputs.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
        return name

    def add_output(self, name: str, shape: tuple[int]) -> str:
        self._outputs.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
        return name

    def add_matmul(
        self, input: str, shape: tuple[int], output: Optional[str] = None, data: Optional[np.ndarray] = None
    ) -> str:
        i = len(self._nodes)

        w_name = f"W_{i}"
        if data is None:
            w_values = np.random.rand(*shape).astype(np.float32)
        else:
            w_values = data
        w_initializer = onnx.helper.make_tensor(
            name=w_name, data_type=onnx.TensorProto.FLOAT, dims=shape, vals=w_values.tobytes(), raw=True
        )
        self._initializers.append(w_initializer)

        output = f"MatMul_{i}_output" if output is None else output
        self._nodes.append(
            onnx.helper.make_node(op_type="MatMul", inputs=[input, w_name], outputs=[output], name=f"MatMul_{i}")
        )
        return output

    def add_mul(self, input_a: str, input_b: str, output: Optional[str] = None) -> str:
        i = len(self._nodes)

        output = f"Mul_{i}_output" if output is None else output
        self._nodes.append(
            onnx.helper.make_node(op_type="Mul", inputs=[input_a, input_b], outputs=[output], name=f"Mul_{i}")
        )
        return output

    def add_relu(self, input: str, output: Optional[str] = None) -> str:
        i = len(self._nodes)

        output = f"Relu_{i}_output" if output is None else output
        self._nodes.append(onnx.helper.make_node(op_type="Relu", inputs=[input], outputs=[output], name=f"Relu_{i}"))
        return output

    def add_selu(self, input: str, output: Optional[str] = None) -> str:
        i = len(self._nodes)

        output = f"Selu_{i}_output" if output is None else output
        self._nodes.append(onnx.helper.make_node(op_type="Selu", inputs=[input], outputs=[output], name=f"Selu_{i}"))
        return output

    def add_unsqueeze(self, input: str, axes: tuple[int, ...], output: Optional[str] = None) -> str:
        i = len(self._nodes)

        axes_name = "Unsqueeze_{i}_axes"
        axes_data = np.array(axes, dtype=np.int64)
        axes_initializer = onnx.helper.make_tensor(
            name=axes_name,
            data_type=onnx.helper.np_dtype_to_tensor_dtype(axes_data.dtype),
            dims=axes_data.shape,
            vals=axes_data.tobytes(),
            raw=True,
        )
        self._initializers.append(axes_initializer)

        output = f"Unsqueeze_{i}_output" if output is None else output
        self._nodes.append(
            onnx.helper.make_node(
                op_type="Unsqueeze", inputs=[input, axes_name], outputs=[output], name=f"Unsqueeze_{i}"
            )
        )
        return output

    def add_cos(self, input: str, output: Optional[str] = None) -> str:
        i = len(self._nodes)

        output = f"Cos_{i}_output" if output is None else output
        self._nodes.append(onnx.helper.make_node(op_type="Cos", inputs=[input], outputs=[output], name=f"Cos_{i}"))
        return output

    def add_sin(self, input: str, output: Optional[str] = None) -> str:
        i = len(self._nodes)

        output = f"Sin_{i}_output" if output is None else output
        self._nodes.append(onnx.helper.make_node(op_type="Sin", inputs=[input], outputs=[output], name=f"Sin_{i}"))
        return output

    def add_transpose(self, input: str, perm: list[int], output: Optional[str] = None) -> str:
        i = len(self._nodes)

        output = f"Transpose_{i}_output" if output is None else output
        self._nodes.append(
            onnx.helper.make_node(
                op_type="Transpose", inputs=[input], outputs=[output], name=f"Transpose_{i}", perm=perm
            )
        )
        return output

    def add_concat(self, inputs: list[str], axis: int, output: Optional[str] = None) -> str:
        i = len(self._nodes)

        output = f"Concat_{i}_output" if output is None else output
        self._nodes.append(
            onnx.helper.make_node(op_type="Concat", inputs=inputs, outputs=[output], name=f"Concat_{i}", axis=axis)
        )
        return output

    def build(self, opset_version: int = 13, ir_version: int = 9) -> onnx.ModelProto:
        graph = onnx.helper.make_graph(self._nodes, self._graph_name, self._inputs, self._outputs, self._initializers)

        op = onnx.OperatorSetIdProto()
        op.version = opset_version

        model = onnx.helper.make_model(graph, opset_imports=[op], ir_version=ir_version)
        return model
