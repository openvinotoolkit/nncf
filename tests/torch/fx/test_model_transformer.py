# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
from torch._export import capture_pre_autograd_graph

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout


def test_leaf_module_insertion_transformation():

    class InsertionPointTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 1, 1, 1)
            self.linear_wts = nn.Parameter(torch.FloatTensor(size=(100, 100)))
            self.conv2 = nn.Conv2d(1, 1, 1, 1)
            self.relu = nn.ReLU()

        def forward(self, input_):
            x = self.conv1(input_)
            x = x.flatten()
            x = nn.functional.linear(x, self.linear_wts)
            x = x.reshape((1, 1, 10, 10))
            x = self.conv2(x)
            x = self.relu(x)
            return x

    model = InsertionPointTestModel()

    with torch.no_grad():
        ex_input = torch.ones([1, 1, 10, 10])
        model.eval()
        exported_model = capture_pre_autograd_graph(model, args=(ex_input,))

    from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
    from nncf.experimental.torch.fx.model_transformer import FXModelTransformer
    from nncf.experimental.torch.fx.transformations import leaf_module_insertion_transformation_builder
    from nncf.torch.graph.transformations.commands import PTTargetPoint

    model_transformer = FXModelTransformer(exported_model)

    conv1_node_name = "conv2d"
    target_point = PTTargetPoint(
        target_type=TargetType.OPERATION_WITH_WEIGHTS, target_node_name=conv1_node_name, input_port_id=1
    )
    transformation = leaf_module_insertion_transformation_builder(exported_model, [target_point])
    command = FXApplyTransformationCommand(transformation)
    transformation_layout = TransformationLayout()
    transformation_layout.register(command)
    model_transformer.transform(transformation_layout)
