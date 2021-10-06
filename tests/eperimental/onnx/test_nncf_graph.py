import onnx
import numpy as np

from nncf.experimental.onnx.nncf_network import NNCFNetwork
from nncf.experimental.onnx.quantization.algorithm import apply_quantization


# def accuracy(output_tensor, target):
#     pred_class = output_tensor.argsort()[::-1]
#     print (pred_class)
#     res = pred_class == target
#     return np.sum(res) / res.size
#
#
# def validate_onnx(onnx_model_path, val_loader):
#     import onnxruntime as rt
#     import numpy as np
#     sess = rt.InferenceSession(onnx_model_path)
#     input_name = sess.get_inputs()[0].name
#     avg_acc1 = 0
#     cnt = 0
#     for i, (input_, target) in enumerate(val_loader):
#         input_tensor = input_.cpu().detach().numpy()
#         target_tensor = target.cpu().detach().numpy()
#
#         output_tensor = sess.run([], {input_name: input_tensor.astype(np.float32)})[0]
#
#         print(target_tensor)
#
#         acc1 = accuracy(output_tensor, target_tensor)
#         print (f'acc1 = {acc1}')
#         avg_acc1 += acc1
#         cnt = i
#         if i == 2:
#             break
#     return avg_acc1 / cnt


def test_nncf_graph():
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

    nncf_network = NNCFNetwork(onnx_model)
    apply_quantization(nncf_network, train_loader, 10)

    # acc1 = validate_onnx('/home/aleksei/nncf_work/onnx_quantization/resnet50.onnx', val_loader)
    # print (f'Average acc1 = {acc1}')