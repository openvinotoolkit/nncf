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
import torch
from torch import nn


class SmallBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        identity_data = x
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        output = torch.add(output, identity_data)
        return output


class SmallModel(nn.Module):
    def __init__(self, scale=3, num_of_ch_enc=16, num_of_ch_dec=8, num_of_res_blocks=4):
        super().__init__()

        self.conv_input = nn.Conv2d(
            in_channels=3, out_channels=num_of_ch_enc, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_cubic1 = nn.Conv2d(
            in_channels=3, out_channels=num_of_ch_dec, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv_cubic2 = nn.Conv2d(
            in_channels=num_of_ch_dec, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.residual1 = SmallBlock(num_of_ch_enc)
        self.residual2 = SmallBlock(num_of_ch_enc)
        self.residual3 = SmallBlock(num_of_ch_enc)
        self.residual4 = SmallBlock(num_of_ch_enc)

        self.conv_mid = nn.Conv2d(
            in_channels=num_of_ch_enc * (num_of_res_blocks + 1),
            out_channels=num_of_ch_dec,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        if scale == 4:
            factor = 2
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_of_ch_dec,
                    out_channels=num_of_ch_dec * factor * factor,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    num_of_ch_dec, num_of_ch_dec * factor * factor, kernel_size=3, padding=1, stride=1, bias=True
                ),
                nn.PixelShuffle(factor),
                nn.ReLU(inplace=True),
            )
        elif scale == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_of_ch_dec,
                    out_channels=num_of_ch_dec * scale * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.ReLU(inplace=True),
            )
        else:
            raise NotImplementedError

        self.conv_output = nn.Conv2d(
            in_channels=num_of_ch_dec, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, x):
        input_ = x[0]
        cubic = x[1]

        c1 = self.conv_cubic1(cubic)
        c1 = self.relu(c1)
        c2 = self.conv_cubic2(c1)
        c2 = self.sigmoid(c2)

        in1 = self.conv_input(input_)

        out = self.relu(in1)

        out1 = self.residual1(out)
        out2 = self.residual2(out1)
        out3 = self.residual3(out2)
        out4 = self.residual4(out3)

        out = torch.cat([out, out1, out2, out3, out4], dim=1)
        out = self.conv_mid(out)
        out = self.relu(out)
        out = self.upscale(out)
        out = self.conv_output(out)

        return [torch.add(out * c2, cubic)]
