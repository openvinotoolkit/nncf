import os

import torchvision
from torchvision import transforms

import onnx

from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantization
from nncf.experimental.post_training.utils import export

from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantizationParameters
from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithms
from nncf.experimental.post_training.initialization.quantizer_range_finder import QuantizerRangeFinderParameters


def create_train_dataset(dataset_dir):
    image_size = [224, 224]
    size = int(image_size[0] / 0.875)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform)


fp32_model_onnx_path = '/home/aleksei/nncf_work/onnx_quantization/mobilenet_v2.onnx'
int8_model_onnx_path = '/home/aleksei/tmp/onnx/onnx_ptq_api/transformed.onnx'
dataset_dir = '/home/aleksei/datasetsimagenet'

fp32_model = onnx.load(fp32_model_onnx_path)

# Step 1: Wrap a model.
compressed_model = CompressedModel(fp32_model)

# Step 2: Initialize the data loader.
train_dataset = create_train_dataset(dataset_dir)


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


dataloader = MyDataLoader(train_dataset, batch_size=1, shuffle=True)

# Step 3: Create a pipeline of compression algorithms.
builder = CompressionBuilder()
# Step 4: Create a quantiztion algorithm.
quantization_parameters = PostTrainingQuantizationParameters(
    iterations_number=10
)
quantization = PostTrainingQuantization(quantization_parameters)
builder.add_algorithm(quantization)
# Step 4: Execute the pipeline.
builder.apply(compressed_model, dataloader)
# Step 5: Export the compressed model.
export(compressed_model, int8_model_onnx_path)
