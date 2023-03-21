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
import json
from pathlib import Path
from typing import Tuple

import openvino.runtime as ov
import torch
from torchvision import datasets
from torchvision import transforms

import nncf
from nncf.experimental.openvino_native.activation_sparsity_statistic.activation_sparsity_statistic import \
    estimate_activation_sparsity


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Activation sparsity statistic for ResNet models.",
        description="Collect activation sparsity statistic.",
    )
    parser.add_argument("-m", "--model_path", help="Target OpenVINO model (.xml).", required=True)
    parser.add_argument("-d", "--data_path", help="Path to folder with images.", required=True)
    parser.add_argument("-s", "--subset_size", help="Size of subset of dataset", default=100)
    parser.add_argument(
        "-t",
        "--target_types",
        help="List of nodes for which statistics will be collected. To collect for all nodes set arguments as 'all'.",
        default=None,
    )
    return parser.parse_args()


def run_example(model_path: str, data_path: str):
    # Step 1: Initialize OpenVINO model.
    ir_model_xml = Path(model_path)
    ov_model = ov.Core().read_model(ir_model_xml)

    # Step 2: Create dataset.
    dataset = create_dataset(data_path, tuple(ov_model.inputs[0].shape)[2:])

    # Step 3: Collect activation sparsity statistics.
    target_types = args.target_types.replace(" ", "").split(",") if args.target_types else None
    activation_sparsity_dict = estimate_activation_sparsity(
        model=ov_model,
        dataset=dataset,
        subset_size=args.subset_size,
        target_node_types=target_types,
        threshold=0.2,
    )


    # Step 4: Save modified model.
    json_path = ir_model_xml.parent / "activation_sparsity_statistics.json"
    with json_path.open("w", encoding="UTF-8") as target:
        json.dump(activation_sparsity_dict, target)
    print(f"Activation sparsity statistics saved to {json_path.as_posix()}")


def create_dataset(data_path: str, img_size: Tuple[int, int]) -> torch.utils.data.DataLoader:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(
        root=f"{data_path}",
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
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
