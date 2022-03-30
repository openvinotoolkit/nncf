from nncf.experimental.onnx.dataloaders.imagenet_dataloader import create_binarized_imagenet_dataset
from nncf.experimental.onnx.dataloaders.imagenet_dataloader import create_dataloader_from_imagenet_torch_dataset
from nncf.experimental.onnx.dataloaders.imagenet_dataloader import binarize_imagenet_dataset


def test_binarize_dataset():
    # dataloader = create_dataloader_from_imagenet_torch_dataset('/home/aleksei/datasetsimagenet/val', [1,3,224,224])
    # binarize_imagenet_dataset(dataloader)
    imagenet_binary_path = '/home/aleksei/tmp/imagenet_binary'
    import numpy as np
    arr = np.load(imagenet_binary_path + '/0.npy', allow_pickle=True)


    dataloader = create_binarized_imagenet_dataset(imagenet_binary_path)
    for sample in dataloader:
        print(sample)
