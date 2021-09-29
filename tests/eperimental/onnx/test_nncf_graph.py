import onnx

from nncf.experimental.onnx.nncf_network import NNCFNetwork
from nncf.experimental.onnx.quantization.algorithm import apply_quantization


def test_nncf_graph():
    onnx_model = onnx.load('/home/aleksei/nncf_work/onnx_quantization/resnet50.onnx')

    from examples.torch.classification.main import create_datasets, create_data_loaders
    from examples.torch.common.sample_config import SampleConfig
    # Data loading code
    config = SampleConfig.from_json('/home/aleksei/nncf_work/nncf_pytorch/examples/torch/classification/configs/quantization/inception_v3_imagenet_int8.json')
    config.update({'dataset': 'imagenet'})
    config.update({'workers': 1})
    config.update({'dataset_dir': '/mnt/icv_externalN/omz-training-datasets/imagenet/'})

    train_dataset, val_dataset = create_datasets(config)
    train_loader, train_sampler, val_loader, init_loader = create_data_loaders(config, train_dataset, val_dataset)

    nncf_network = NNCFNetwork(onnx_model)
    apply_quantization(nncf_network, train_loader)




    # stats_for_range_init = self._get_statistics_for_final_range_init(target_model,
    #                                                                  self._single_config_quantizer_setup,
    #                                                                  self._range_init_params)
    # minmax_values_for_range_init = self._get_minmax_values_for_quantizer_locations(
    #     self._single_config_quantizer_setup,
    #     stats_for_range_init,
    #     target_model_graph)
    # final_setup = self._handle_quantize_inputs_option(final_setup, nncf_graph)