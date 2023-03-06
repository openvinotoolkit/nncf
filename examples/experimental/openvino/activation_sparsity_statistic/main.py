"""
 Copyright (c) 2023 Intel Corporation
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
import argparse
from pathlib import Path

import openvino.runtime as ov
import torch
from torchvision import datasets
from torchvision import transforms

import nncf
from nncf.experimental.openvino_native.activation_sparsity_statistic.activation_sparsity_statistic import \
    activation_sparsity_statistic_impl


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Activation sparsity statistic for ResNet models.",
        description="Collect activation sparsity statistic and write statistic to the model.",
    )
    parser.add_argument("-m", "--model_path", help="Target OpenVINO model (.xml).")
    parser.add_argument("-d", "--data_path", help="Path to folder with images.")
    return parser.parse_args()


def run_example(model_path, data_path):
    # Step 1: Initialize OpenVINO model.
    ir_model_xml = Path(model_path)
    ir_model_bin = ir_model_xml.with_suffix(".bin")
    ov_model = ov.Core().read_model(ir_model_xml, ir_model_bin)

    # Step 2: Create dataset.
    dataset = create_dataset(data_path)

    # Step 3: Collect activation sparsity statistics.
    modified_model = activation_sparsity_statistic_impl(
        model=ov_model, dataset=dataset, subset_size=100, target_node_types=None, threshold=0.05
    )

    # Step 4: Save modified model.
    new_model_path = ir_model_xml.with_name(f"modified_{ir_model_xml.name}")
    ov.serialize(modified_model, new_model_path.as_posix())
    print(f"Model saved: {new_model_path.as_posix()}")


def create_dataset(data_path) -> torch.utils.data.DataLoader:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(
        root=f"{data_path}",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    def transform_fn(data_item):
        images, _ = data_item
        return images

    return nncf.Dataset(loader, transform_fn)


if __name__ == "__main__":
    args = parse_args()
    run_example(args.model_path, args.data_path)
