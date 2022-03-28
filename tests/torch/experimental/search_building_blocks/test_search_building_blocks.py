"""
 Copyright (c) 2021 Intel Corporation
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
from functools import partial

import pytest
from torchvision.models import MobileNetV2

from examples.torch.common.models import efficient_net
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from tests.torch import test_models
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.test_elastic_depth import INCEPTION_INPUT_SIZE
from tests.torch.nas.test_elastic_depth import RESNET50_INPUT_SIZE
from tests.torch.test_models import ResNet18
from tests.torch.test_models import squeezenet1_0
from tests.torch.test_models.inceptionv3 import Inception3
from tests.torch.test_models.resnet import ResNet50

REF_BUILDING_BLOCKS_FOR_RESNET = [
    BuildingBlock('ResNet/Sequential[layer1]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer1]/Bottleneck[1]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer1]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer1]/Bottleneck[2]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer2]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer2]/Bottleneck[1]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer2]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer2]/Bottleneck[2]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer2]/Bottleneck[2]/relu_2', 'ResNet/Sequential[layer2]/Bottleneck[3]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[1]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[2]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[2]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[3]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[3]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[4]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer3]/Bottleneck[4]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[5]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer4]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer4]/Bottleneck[1]/relu_2'),
    BuildingBlock('ResNet/Sequential[layer4]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer4]/Bottleneck[2]/relu_2')
]

REF_BUILDING_BLOCKS_FOR_MOBILENETV2 = [
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[3]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[5]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[5]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[6]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[8]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[8]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[9]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[9]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[10]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[12]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[12]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[13]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/NNCFBatchNorm2d[3]/batch_norm_0',
        'MobileNetV2/Sequential[features]/InvertedResidual[15]/__add___0'),
    BuildingBlock(
        'MobileNetV2/Sequential[features]/InvertedResidual[15]/__add___0',
        'MobileNetV2/Sequential[features]/InvertedResidual[16]/__add___0')
]

REF_BUILDING_BLOCKS_FOR_SQUEEZENET = [
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[3]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/relu__0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[4]/cat_0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[7]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[7]/cat_0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[8]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/relu__0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[9]/cat_0'),
    BuildingBlock('SqueezeNet/Sequential[features]/Fire[12]/ReLU[squeeze_activation]/relu__0',
                  'SqueezeNet/Sequential[features]/Fire[12]/cat_0')
]

REF_BUILDING_BLOCKS_FOR_INCEPTIONV3 = [
    BuildingBlock('Inception3/InceptionA[Mixed_5c]/cat_0',
                  'Inception3/InceptionA[Mixed_5d]/cat_0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionB[Mixed_6a]/cat_0',
                  'Inception3/InceptionC[Mixed_6b]/cat_0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6b]/cat_0',
                  'Inception3/InceptionC[Mixed_6c]/cat_0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6c]/cat_0',
                  'Inception3/InceptionC[Mixed_6d]/cat_0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_4]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_3]/relu__0',
                  'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_5]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_2]/relu__0',
                  'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_5]/relu__0'),
    BuildingBlock('Inception3/InceptionC[Mixed_6d]/cat_0',
                  'Inception3/InceptionC[Mixed_6e]/cat_0'),
    BuildingBlock('Inception3/InceptionE[Mixed_7b]/cat_2',
                  'Inception3/InceptionE[Mixed_7c]/cat_2')
]

REF_BUILDING_BLOCKS_FOR_ResNext = [
    BuildingBlock('ResNeXt/Sequential[layer1]/Block[0]/relu_2', 'ResNeXt/Sequential[layer1]/Block[1]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer1]/Block[1]/relu_2', 'ResNeXt/Sequential[layer1]/Block[2]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer2]/Block[0]/relu_2', 'ResNeXt/Sequential[layer2]/Block[1]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer2]/Block[1]/relu_2', 'ResNeXt/Sequential[layer2]/Block[2]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer3]/Block[0]/relu_2', 'ResNeXt/Sequential[layer3]/Block[1]/relu_2'),
    BuildingBlock('ResNeXt/Sequential[layer3]/Block[1]/relu_2', 'ResNeXt/Sequential[layer3]/Block[2]/relu_2')]

REF_BUILDING_BLOCKS_FOR_PNASNetB = [
    BuildingBlock('PNASNet/relu_0', 'PNASNet/Sequential[layer1]/CellB[0]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[0]/relu_2', 'PNASNet/Sequential[layer1]/CellB[1]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[1]/relu_2', 'PNASNet/Sequential[layer1]/CellB[2]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[2]/relu_2', 'PNASNet/Sequential[layer1]/CellB[3]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[3]/relu_2', 'PNASNet/Sequential[layer1]/CellB[4]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer1]/CellB[4]/relu_2', 'PNASNet/Sequential[layer1]/CellB[5]/relu_2'),
    BuildingBlock('PNASNet/CellB[layer2]/relu_2', 'PNASNet/Sequential[layer3]/CellB[0]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[0]/relu_2', 'PNASNet/Sequential[layer3]/CellB[1]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[1]/relu_2', 'PNASNet/Sequential[layer3]/CellB[2]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[2]/relu_2', 'PNASNet/Sequential[layer3]/CellB[3]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[3]/relu_2', 'PNASNet/Sequential[layer3]/CellB[4]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer3]/CellB[4]/relu_2', 'PNASNet/Sequential[layer3]/CellB[5]/relu_2'),
    BuildingBlock('PNASNet/CellB[layer4]/relu_2', 'PNASNet/Sequential[layer5]/CellB[0]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[0]/relu_2', 'PNASNet/Sequential[layer5]/CellB[1]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[1]/relu_2', 'PNASNet/Sequential[layer5]/CellB[2]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[2]/relu_2', 'PNASNet/Sequential[layer5]/CellB[3]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[3]/relu_2', 'PNASNet/Sequential[layer5]/CellB[4]/relu_2'),
    BuildingBlock('PNASNet/Sequential[layer5]/CellB[4]/relu_2', 'PNASNet/Sequential[layer5]/CellB[5]/relu_2')]

REF_BUILDING_BLOCKS_FOR_SSD_MOBILENET = [
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[7]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[8]/ReLU[2]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[7]/ReLU[5]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[8]/ReLU[5]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[7]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[8]/ReLU[5]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[8]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[2]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[7]/ReLU[5]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[2]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[8]/ReLU[5]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[5]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[8]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[5]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[10]/ReLU[2]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[8]/ReLU[5]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[10]/ReLU[2]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[5]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[10]/ReLU[5]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[10]/ReLU[5]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[10]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[11]/ReLU[2]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[9]/ReLU[5]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[11]/ReLU[2]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[10]/ReLU[5]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[11]/ReLU[5]/relu__0'),
    BuildingBlock('MobileNetSSD/MultiOutputSequential[basenet]/Sequential[10]/ReLU[2]/relu__0',
                  'MobileNetSSD/MultiOutputSequential[basenet]/Sequential[11]/ReLU[5]/relu__0'), ]

REF_BUILDING_BLOCKS_FOR_EFFICIENT_NET = [
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[0]/MemoryEfficientSwish[_swish]/__mul___0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[0]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[1]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[1]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[2]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[2]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[1]/NNCFBatchNorm2d[_bn2]/batch_norm_0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[2]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[3]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[3]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[4]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[4]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[3]/NNCFBatchNorm2d[_bn2]/batch_norm_0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[4]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[5]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[5]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[5]/NNCFBatchNorm2d[_bn2]/batch_norm_0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[7]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[7]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[6]/__add___0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[7]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[8]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[8]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[8]/NNCFBatchNorm2d[_bn2]/batch_norm_0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[10]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[10]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[9]/__add___0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[10]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[11]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[11]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[11]/NNCFBatchNorm2d[_bn2]/batch_norm_0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[12]/__add___0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[14]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[14]/__mul___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[13]/__add___0',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[14]/__add___0'),
    BuildingBlock('EfficientNet/ModuleList[_blocks]/MBConvBlock[15]/MemoryEfficientSwish[_swish]/__mul___1',
                  'EfficientNet/ModuleList[_blocks]/MBConvBlock[15]/__mul___0')]

REF_GROUP_DEPENDENT_RESNET50 = {0: [0, 1], 1: [2, 3, 4], 2: [5, 6, 7, 8, 9], 3: [10, 11]}
REF_GROUP_DEPENDENT_MOBILENETV2 = {0: [0], 1: [1, 2], 2: [3, 4, 5], 3: [6, 7], 4: [8, 9]}
REF_GROUP_DEPENDENT_INCEPTIONV3 = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9],
                                   10: [10], 11: [11]}
REF_GROUP_DEPENDENT_SQUEEZNET = {0: [0, 1], 1: [2], 2: [3, 4], 3: [5]}
REF_GROUP_DEPENDENT_PNASNETB = {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10, 11], 2: [12, 13, 14, 15, 16, 17]}
REF_GROUP_DEPENDENT_RESNEXT = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
REF_GROUP_DEPENDENT_SSD_MOBILENET = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9],
                                     10: [10], 11: [11], 12: [12], 13: [13], 14: [14]}
REF_GROUP_DEPENDENT_EFFICIENT_NET = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9],
                                     10: [10], 11: [11], 12: [12], 13: [13], 14: [14], 15: [15], 16: [16], 17: [17],
                                     18: [18], 19: [19], 20: [20], 21: [21], 22: [22], 23: [23], 24: [24]}


@pytest.mark.parametrize('model_creator, input_sizes, ref_skipped_blocks, ref_group_dependent',
                         ((ResNet50, RESNET50_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_RESNET, REF_GROUP_DEPENDENT_RESNET50),
                          (MobileNetV2, RESNET50_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_MOBILENETV2,
                           REF_GROUP_DEPENDENT_MOBILENETV2),
                          (Inception3, INCEPTION_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_INCEPTIONV3,
                           REF_GROUP_DEPENDENT_INCEPTIONV3),
                          (squeezenet1_0, RESNET50_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_SQUEEZENET,
                           REF_GROUP_DEPENDENT_SQUEEZNET),
                          (test_models.ResNeXt29_32x4d, [1, 3, 32, 32], REF_BUILDING_BLOCKS_FOR_ResNext,
                           REF_GROUP_DEPENDENT_RESNEXT),
                          (test_models.PNASNetB, [1, 3, 32, 32], REF_BUILDING_BLOCKS_FOR_PNASNetB,
                           REF_GROUP_DEPENDENT_PNASNETB),
                          (test_models.ssd_mobilenet, [2, 3, 300, 300], REF_BUILDING_BLOCKS_FOR_SSD_MOBILENET,
                           REF_GROUP_DEPENDENT_SSD_MOBILENET),
                          (partial(efficient_net, model_name='efficientnet-b0'), [10, 3, 240, 240],
                           REF_BUILDING_BLOCKS_FOR_EFFICIENT_NET, REF_GROUP_DEPENDENT_EFFICIENT_NET)
                          ))
def test_building_block(model_creator, input_sizes, ref_skipped_blocks, ref_group_dependent):
    model = model_creator()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=input_sizes)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    blocks, _, group_dependent = get_building_blocks(compressed_model, allow_nested_blocks=False)
    assert blocks == ref_skipped_blocks
    assert group_dependent == ref_group_dependent


class SearchBBlockAlgoParamsCase:
    def __init__(self,
                 min_block_size: int = 1,
                 max_block_size: int = 100,
                 allow_nested_blocks: bool = False,
                 allow_linear_combination: bool = True,
                 ref_blocks=None):
        self.max_block_size = max_block_size
        self.min_block_size = min_block_size
        self.allow_nested_blocks = allow_nested_blocks
        self.allow_linear_combination = allow_linear_combination
        self.ref_blocks = [] if ref_blocks is None else ref_blocks


@pytest.mark.parametrize('algo_params', (
    (
        SearchBBlockAlgoParamsCase(max_block_size=7,
                                   ref_blocks=[BuildingBlock("ResNet/MaxPool2d[maxpool]/max_pool2d_0",
                                                             "ResNet/Sequential[layer1]/BasicBlock[0]/relu_1")]),
        SearchBBlockAlgoParamsCase(max_block_size=8,
                                   ref_blocks=[BuildingBlock("ResNet/MaxPool2d[maxpool]/max_pool2d_0",
                                                             "ResNet/Sequential[layer1]/BasicBlock[0]/relu_1"),
                                               BuildingBlock("ResNet/Sequential[layer1]/BasicBlock[0]/relu_1",
                                                             "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1"),
                                               BuildingBlock("ResNet/Sequential[layer2]/BasicBlock[0]/relu_1",
                                                             "ResNet/Sequential[layer2]/BasicBlock[1]/relu_1"),
                                               BuildingBlock("ResNet/Sequential[layer3]/BasicBlock[0]/relu_1",
                                                             "ResNet/Sequential[layer3]/BasicBlock[1]/relu_1"),
                                               BuildingBlock("ResNet/Sequential[layer4]/BasicBlock[0]/relu_1",
                                                             "ResNet/Sequential[layer4]/BasicBlock[1]/relu_1")
                                               ]),
        SearchBBlockAlgoParamsCase(min_block_size=10,
                                   max_block_size=20,
                                   ref_blocks=[BuildingBlock("ResNet/MaxPool2d[maxpool]/max_pool2d_0",
                                                             "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1")]),
        SearchBBlockAlgoParamsCase(max_block_size=20,
                                   allow_nested_blocks=True,
                                   ref_blocks=[  # start nested block group
                                       BuildingBlock("ResNet/MaxPool2d[maxpool]/max_pool2d_0",
                                                     "ResNet/Sequential[layer1]/BasicBlock[0]/relu_1"),  # block A
                                       BuildingBlock("ResNet/Sequential[layer1]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1"),  # block B
                                       BuildingBlock("ResNet/MaxPool2d[maxpool]/max_pool2d_0",
                                                     "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1"),  # block C
                                       # end nested block group
                                       # C = A + B . A, B - nested blocks. Lin. combination A + B = block C
                                       BuildingBlock("ResNet/Sequential[layer2]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer2]/BasicBlock[1]/relu_1"),
                                       BuildingBlock("ResNet/Sequential[layer3]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer3]/BasicBlock[1]/relu_1"),
                                       BuildingBlock("ResNet/Sequential[layer4]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer4]/BasicBlock[1]/relu_1")]),
        SearchBBlockAlgoParamsCase(max_block_size=20,
                                   allow_nested_blocks=True,
                                   allow_linear_combination=False,
                                   ref_blocks=[
                                       # nested block group is empty because parameter allow_linear_combination is false
                                       # and the biggest block was deleted.
                                       BuildingBlock("ResNet/MaxPool2d[maxpool]/max_pool2d_0",
                                                     "ResNet/Sequential[layer1]/BasicBlock[0]/relu_1"),
                                       BuildingBlock("ResNet/Sequential[layer1]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer1]/BasicBlock[1]/relu_1"),
                                       BuildingBlock("ResNet/Sequential[layer2]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer2]/BasicBlock[1]/relu_1"),
                                       BuildingBlock("ResNet/Sequential[layer3]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer3]/BasicBlock[1]/relu_1"),
                                       BuildingBlock("ResNet/Sequential[layer4]/BasicBlock[0]/relu_1",
                                                     "ResNet/Sequential[layer4]/BasicBlock[1]/relu_1")]))
))
def test_building_block_algo_param(algo_params: SearchBBlockAlgoParamsCase):
    model = ResNet18()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    blocks, _, _ = get_building_blocks(compressed_model,
                                       allow_nested_blocks=algo_params.allow_nested_blocks,
                                       min_block_size=algo_params.min_block_size,
                                       max_block_size=algo_params.max_block_size,
                                       allow_linear_combination=algo_params.allow_linear_combination)

    assert blocks == algo_params.ref_blocks
