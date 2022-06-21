"""
 Copyright (c) 2022 Intel Corporation
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

import onnx
from onnx import version_converter  # pylint: disable=no-name-in-module
from nncf.common.utils.logger import logger as nncf_logger

class ONNNXModelNormalizer:
    @staticmethod
    def modify_onnx_model_for_quantization(model: onnx.ModelProto) -> onnx.ModelProto:
        onnx.checker.check_model(model)
        nncf_logger.info('Original opset = {}'.format(model.opset_import[0].version))
        nncf_logger.info('Original ir_version = {}'.format(model.ir_version))

        model.ir_version = 7  # Due to the 'Shufflenet-v1
        modified_model = version_converter.convert_version(model, 13)
        # ONNX shape inference
        # https://github.com/onnx/onnx/blob/main/docs/proposals/SymbolicShapeInfProposal.md
        modified_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(modified_model)
        onnx.checker.check_model(modified_model)
        nncf_logger.info(
            'Successfully converted the model to the opset = {}'.format(modified_model.opset_import[0].version))

        for i, node in enumerate(modified_model.graph.node):
            if node.name == '':
                node.name = node.op_type + '_nncf_' + str(i)
        return modified_model
