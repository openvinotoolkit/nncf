"""
 Copyright (c) 2020 Intel Corporation
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
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter, Dropout


class ManyNonEvalModules(nn.Module):
    class AuxBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self.weight = Parameter(torch.ones([1, 1]))

        def forward(self, x):
            x = F.linear(x, self.weight)
            x = self.linear(x)
            x = F.relu(x)
            return x

    class ModuleWithMixedModules(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(torch.ones([1, 1]))
            self.not_called_linear = nn.Linear(1, 1)
            self.called_linear = nn.Linear(1, 1)

        def forward(self, x):
            x = Dropout(p=0.2)(x)
            x = F.linear(x, self.weight)
            x = Dropout(p=0.2)(x)
            x = self.called_linear(x)
            return x

    def __init__(self):
        super().__init__()
        self.dummy = Parameter(torch.ones([1]))
        self.aux_branch = self.AuxBranch()
        self.mixed_modules = self.ModuleWithMixedModules()
        self.avg_pool = nn.AvgPool2d(1)

    def forward(self, x):
        x = self.avg_pool(x)
        if self.training:
            aux = self.aux_branch(x)
        x = self.mixed_modules(x)
        return x, aux if self.training else x
