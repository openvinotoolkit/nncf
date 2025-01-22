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

from pathlib import Path
from typing import Tuple, Union

import torch.fx
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from fastdownload import FastDownload
from torch.fx.passes.graph_drawer import FxGraphDrawer

from nncf.experimental.torch.fx.transformations import apply_quantization_transformations
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTConstNoopMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTSumMetatype
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation


class TinyImagenetDatasetManager:
    DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    DATASET_PATH = "~/.cache/nncf/tests/datasets"

    def __init__(self, image_size: int, batch_size: int) -> None:
        self.image_size = image_size
        self.batch_size = batch_size

    @staticmethod
    def download_dataset() -> Path:
        downloader = FastDownload(base=TinyImagenetDatasetManager.DATASET_PATH, archive="downloaded", data="extracted")
        return downloader.get(TinyImagenetDatasetManager.DATASET_URL)

    @staticmethod
    def prepare_tiny_imagenet_200(dataset_dir: Path):
        # Format validation set the same way as train set is formatted.
        val_data_dir = dataset_dir / "val"
        val_images_dir = val_data_dir / "images"
        if not val_images_dir.exists():
            return

        val_annotations_file = val_data_dir / "val_annotations.txt"
        with open(val_annotations_file, "r") as f:
            val_annotation_data = map(lambda line: line.split("\t")[:2], f.readlines())
        for image_filename, image_label in val_annotation_data:
            from_image_filepath = val_images_dir / image_filename
            to_image_dir = val_data_dir / image_label
            if not to_image_dir.exists():
                to_image_dir.mkdir()
            to_image_filepath = to_image_dir / image_filename
            from_image_filepath.rename(to_image_filepath)
        val_annotations_file.unlink()
        val_images_dir.rmdir()

    def create_data_loaders(self):
        dataset_path = TinyImagenetDatasetManager.download_dataset()

        TinyImagenetDatasetManager.prepare_tiny_imagenet_200(dataset_path)
        print(f"Successfully downloaded and prepared dataset at: {dataset_path}")

        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, sampler=None
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        # Creating separate dataloader with batch size = 1
        # as dataloaders with batches > 1 are not supported yet.
        calibration_dataset = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )

        return train_loader, val_loader, calibration_dataset


def visualize_fx_model(model: torch.fx.GraphModule, output_svg_path: str):
    g = FxGraphDrawer(model, output_svg_path)
    g.get_dot_graph().write_svg(output_svg_path)


def get_torch_fx_model(
    model: torch.nn.Module, ex_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> torch.fx.GraphModule:
    """
    Converts given module to GraphModule.

    :param model: Given torch Module.
    :return: Exported GraphModule.
    """
    try:
        named_param = next(model.named_parameters())
    except StopIteration:
        named_param = None
    if named_param is None:
        device = torch.device("cpu")
    else:
        device = named_param[1].device

    if isinstance(ex_input, torch.Tensor):
        ex_input = (ex_input,)
    device_ex_input = []
    for inp in ex_input:
        device_ex_input.append(inp.to(device))
    device_ex_input = tuple(device_ex_input)

    model.eval()
    with torch.no_grad():
        with disable_patching():
            return torch.export.export_for_training(model, args=device_ex_input).module()


def get_torch_fx_model_q_transformed(model: torch.nn.Module, ex_input: torch.Tensor) -> torch.fx.GraphModule:
    """
    Converts given module to GraphModule and applies required quantization transformations to it.

    :param model: Given torch Module.
    :return: Exported GraphModule.
    """
    fx_model = get_torch_fx_model(model, ex_input)
    apply_quantization_transformations(fx_model)
    return fx_model


def get_single_conv_nncf_graph() -> NNCFGraphToTest:
    return NNCFGraphToTest(
        conv_metatype=PTModuleConv2dMetatype,
        nncf_graph_cls=PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
    )


def get_depthwise_conv_nncf_graph() -> NNCFGraphToTestDepthwiseConv:
    return NNCFGraphToTestDepthwiseConv(
        depthwise_conv_metatype=PTModuleDepthwiseConv2dSubtype,
        nncf_graph_cls=PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
    )


def get_sum_aggregation_nncf_graph() -> NNCFGraphToTestSumAggregation:
    return NNCFGraphToTestSumAggregation(
        conv_metatype=PTModuleConv2dMetatype,
        sum_metatype=PTSumMetatype,
        nncf_graph_cls=PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
    )
