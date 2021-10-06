import onnx

from nncf.experimental.onnx.nncf_network import NNCFNetwork
from nncf.experimental.onnx.quantization.algorithm import apply_quantization


def test_onnx_post_training_quantization():
    onnx_model = onnx.load('/home/aleksei/nncf_work/onnx_quantization/resnet50.onnx')

    from examples.torch.classification.main import create_datasets, create_data_loaders
    from examples.torch.common.sample_config import SampleConfig

    # Data loading code
    config = SampleConfig.from_json('/home/aleksei/nncf_work/nncf_pytorch/examples/torch/classification/configs/quantization/resnet50_imagenet_int8.json')
    config.update({'dataset': 'imagenet'})
    config.update({'workers': 1})
    config.update({'batch_size': 1})
    config.update({'dataset_dir': '/home/aleksei/datasetsimagenet'})

    train_dataset, val_dataset = create_datasets(config)
    train_loader, train_sampler, val_loader, init_loader = create_data_loaders(config, train_dataset, val_dataset)
    number_initialization_samples = 10

    nncf_network = NNCFNetwork(onnx_model)
    apply_quantization(nncf_network, train_loader, number_initialization_samples)

    # 1. remove Config
    # 2. nncf_network to apply_post_training_quantization (?)
    # 3. Data loader
    # 4. List of changes to implement the full functionality of post-training quantization / estimates efforts
    # 5. Signed/unsigned support of quantizers chech if min > 0
    # 6. Per-channel export for weights
    # 7. Optimize statistics collection (?)
