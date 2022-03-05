import os

from typing import List

import onnx

from nncf.experimental.post_training.api.dataloader import DataLoader


class ImageNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset
        print(f"The dataloader is built with the data located on  {dataset.root}")

    def __getitem__(self, item):
        tensor, target = self.dataset[item]
        tensor = tensor.cpu().detach().numpy()
        return tensor, target

    def __len__(self):
        return len(self.dataset)


def create_dataloader_from_imagenet_torch_dataset(dataset_dir, input_shape: List[int], batch_size: int = 1,
                                                  shuffle: bool = True):
    import torchvision
    from torchvision import transforms
    image_size = [input_shape[-2], input_shape[-1]]
    size = int(image_size[0] / 0.875)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    initialization_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform)
    return ImageNetDataLoader(initialization_dataset, batch_size, shuffle)


def add_input_from_initializer(model: onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)
