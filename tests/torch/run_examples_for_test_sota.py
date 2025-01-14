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

import sys

from examples.torch.classification import main as cls_main
from examples.torch.object_detection import main as od_main
from examples.torch.semantic_segmentation import main as seg_main
from tests.torch.helpers import create_dataloader_with_num_workers

cls_main.create_data_loaders = create_dataloader_with_num_workers(
    cls_main.create_data_loaders, num_workers=0, sample_type="classification"
)
od_main.create_dataloaders = create_dataloader_with_num_workers(
    od_main.create_dataloaders, num_workers=0, sample_type="object_detection"
)
seg_main.load_dataset = create_dataloader_with_num_workers(
    seg_main.load_dataset, num_workers=0, sample_type="semantic_segmentation"
)

SAMPLES = {"classification": cls_main.main, "object_detection": od_main.main, "semantic_segmentation": seg_main.main}


def get_main_fn(sample_type):
    return SAMPLES[sample_type]


if __name__ == "__main__":
    main_fn = get_main_fn(sys.argv[1])
    main_fn(sys.argv[2:])
