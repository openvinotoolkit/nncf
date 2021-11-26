import pytest
from torchvision.models import MobileNetV2
from torchvision.models.squeezenet import squeezenet1_0

from nncf.torch.model_creation import create_compressed_model
from tests.torch.helpers import get_empty_config
from nncf.torch.search_blocks import get_building_blocks
from tests.torch.test_models.resnet import ResNet50
from tests.torch.test_models.inceptionv3 import Inception3

RESNET50_INPUT_SIZE = [1, 3, 32, 32]
INCEPTION_INPUT_SIZE = [2, 3, 299, 299]

REF_BUILDING_BLOCKS_FOR_RESNET = [
    ['ResNet/Sequential[layer1]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer1]/Bottleneck[1]/relu_2'],
    ['ResNet/Sequential[layer1]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer1]/Bottleneck[2]/relu_2'],
    ['ResNet/Sequential[layer2]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer2]/Bottleneck[1]/relu_2'],
    ['ResNet/Sequential[layer2]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer2]/Bottleneck[2]/relu_2'],
    ['ResNet/Sequential[layer2]/Bottleneck[2]/relu_2', 'ResNet/Sequential[layer2]/Bottleneck[3]/relu_2'],
    ['ResNet/Sequential[layer3]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[1]/relu_2'],
    ['ResNet/Sequential[layer3]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[2]/relu_2'],
    ['ResNet/Sequential[layer3]/Bottleneck[2]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[3]/relu_2'],
    ['ResNet/Sequential[layer3]/Bottleneck[3]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[4]/relu_2'],
    ['ResNet/Sequential[layer3]/Bottleneck[4]/relu_2', 'ResNet/Sequential[layer3]/Bottleneck[5]/relu_2'],
    ['ResNet/Sequential[layer4]/Bottleneck[0]/relu_2', 'ResNet/Sequential[layer4]/Bottleneck[1]/relu_2'],
    ['ResNet/Sequential[layer4]/Bottleneck[1]/relu_2', 'ResNet/Sequential[layer4]/Bottleneck[2]/relu_2']]  # 12

REF_BUILDING_BLOCKS_FOR_MOBILENETV2 = [
    ['MobileNetV2/Sequential[features]/InvertedResidual[2]/Sequential[conv]/NNCFBatchNorm[3]/batch_norm_0',
     'MobileNetV2/Sequential[features]/InvertedResidual[3]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[4]/Sequential[conv]/NNCFBatchNorm[3]/batch_norm_0',
     'MobileNetV2/Sequential[features]/InvertedResidual[5]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[5]/__add___0',
     'MobileNetV2/Sequential[features]/InvertedResidual[6]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[7]/Sequential[conv]/NNCFBatchNorm[3]/batch_norm_0',
     'MobileNetV2/Sequential[features]/InvertedResidual[8]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[8]/__add___0',
     'MobileNetV2/Sequential[features]/InvertedResidual[9]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[9]/__add___0',
     'MobileNetV2/Sequential[features]/InvertedResidual[10]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[11]/Sequential[conv]/NNCFBatchNorm[3]/batch_norm_0',
     'MobileNetV2/Sequential[features]/InvertedResidual[12]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[12]/__add___0',
     'MobileNetV2/Sequential[features]/InvertedResidual[13]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[14]/Sequential[conv]/NNCFBatchNorm[3]/batch_norm_0',
     'MobileNetV2/Sequential[features]/InvertedResidual[15]/__add___0'],
    ['MobileNetV2/Sequential[features]/InvertedResidual[15]/__add___0',
     'MobileNetV2/Sequential[features]/InvertedResidual[16]/__add___0']]  # 10

REF_BUILDING_BLOCKS_FOR_INCEPTIONV3 = [
    ['Inception3/InceptionA[Mixed_5c]/cat_0', 'Inception3/InceptionA[Mixed_5d]/cat_0'],
    ['Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_2]/relu__0',
     'Inception3/InceptionC[Mixed_6b]/BasicConv2d[branch7x7dbl_4]/relu__0'],
    ['Inception3/InceptionB[Mixed_6a]/cat_0', 'Inception3/InceptionC[Mixed_6b]/cat_0'],
    ['Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_2]/relu__0',
     'Inception3/InceptionC[Mixed_6c]/BasicConv2d[branch7x7dbl_4]/relu__0'],
    ['Inception3/InceptionC[Mixed_6b]/cat_0', 'Inception3/InceptionC[Mixed_6c]/cat_0'],
    ['Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_2]/relu__0',
     'Inception3/InceptionC[Mixed_6d]/BasicConv2d[branch7x7dbl_4]/relu__0'],
    ['Inception3/InceptionC[Mixed_6c]/cat_0', 'Inception3/InceptionC[Mixed_6d]/cat_0'],
    ['Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_2]/relu__0',
     'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_4]/relu__0'],
    ['Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_3]/relu__0',
     'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_5]/relu__0'],
    ['Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_2]/relu__0',
     'Inception3/InceptionC[Mixed_6e]/BasicConv2d[branch7x7dbl_5]/relu__0'],
    ['Inception3/InceptionC[Mixed_6d]/cat_0', 'Inception3/InceptionC[Mixed_6e]/cat_0'],
    ['Inception3/InceptionE[Mixed_7b]/cat_2', 'Inception3/InceptionE[Mixed_7c]/cat_2']]

REF_BUILDING_BLOCKS_FOR_SQUEEZENET1_0 = [
    ['SqueezeNet/Sequential[features]/Fire[3]/ReLU[squeeze_activation]/relu__0',
     'SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/relu__0'],
    ['SqueezeNet/Sequential[features]/Fire[4]/ReLU[squeeze_activation]/relu__0',
     'SqueezeNet/Sequential[features]/Fire[4]/cat_0'],
    ['SqueezeNet/Sequential[features]/Fire[7]/ReLU[squeeze_activation]/relu__0',
     'SqueezeNet/Sequential[features]/Fire[7]/cat_0'],
    ['SqueezeNet/Sequential[features]/Fire[8]/ReLU[squeeze_activation]/relu__0',
     'SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/relu__0'],
    ['SqueezeNet/Sequential[features]/Fire[9]/ReLU[squeeze_activation]/relu__0',
     'SqueezeNet/Sequential[features]/Fire[9]/cat_0'],
    ['SqueezeNet/Sequential[features]/Fire[12]/ReLU[squeeze_activation]/relu__0',
     'SqueezeNet/Sequential[features]/Fire[12]/cat_0']]

@pytest.mark.parametrize('model_creator, input_sizes, ref_building_blocks',
                         ((ResNet50, RESNET50_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_RESNET),
                          (MobileNetV2, RESNET50_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_MOBILENETV2),
                          (Inception3, INCEPTION_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_INCEPTIONV3),
                          (squeezenet1_0, RESNET50_INPUT_SIZE, REF_BUILDING_BLOCKS_FOR_SQUEEZENET1_0)
                           ))

def test_building_block(model_creator, input_sizes, ref_building_blocks):
    model = model_creator()
    nncf_config = get_empty_config(input_sample_sizes=input_sizes)
    _, compressed_model = create_compressed_model(model, nncf_config)

    building_blocks = get_building_blocks(compressed_model)
    assert building_blocks == ref_building_blocks
