"""
 Copyright (c) 2019 Intel Corporation
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
import torch

from nncf import NNCFConfig
from nncf.dynamic_graph.graph_builder import create_input_infos


def compare_multi_gpu_dump(config, dump_dir, get_path_by_rank_fn):
    mismatching = False
    ref_file_path = get_path_by_rank_fn(dump_dir, 0)
    with ref_file_path.open('rb') as ref_scale_file:
        ref_data = torch.load(ref_scale_file)
        for other_rank in range(1, config.world_size):
            other_file_path = get_path_by_rank_fn(dump_dir, other_rank)
            with other_file_path.open('rb') as in_file:
                data_to_compare = torch.load(in_file)
                if ref_data != data_to_compare:
                    mismatching = True
    return mismatching


class RankDatasetMock:
    def __init__(self, input_size, rank):
        self.input_size = input_size
        self.rank = rank
        super().__init__()

    def __getitem__(self, index):
        dummy_input = torch.ones(self.input_size) * (self.rank - 1) * 3
        return dummy_input, torch.ones(1)

    def __len__(self):
        return 100


def get_quantization_config_without_range_init(model_size=4):
    config = NNCFConfig()
    config.update({
        "model": "basic_quant_conv",
        "model_size": model_size,
        "input_info":
            {
                "sample_size": [1, 1, model_size, model_size],
            },
        "compression":
            {
                "algorithm": "quantization",
                "initializer": {
                    "range": {
                        "num_init_steps": 0
                    }
                }
            }
    })
    return config


def get_squeezenet_quantization_config(model_size=32, batch_size=3):
    config = get_quantization_config_without_range_init(model_size)
    config['model'] = 'squeezenet1_1_custom'
    config['input_info'] = {
        "sample_size": [batch_size, 3, model_size, model_size],
    }
    return config


def distributed_init_test_default(gpu, ngpus_per_node, config):
    config.batch_size = 3
    config.workers = 3
    config.gpu = gpu
    config.ngpus_per_node = ngpus_per_node
    config.rank = gpu
    config.distributed = True

    torch.distributed.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:8899',
                                         world_size=config.world_size, rank=config.rank)

    input_infos_list = create_input_infos(config)
    input_sample_size = input_infos_list[0].shape
    data_loader = torch.utils.data.DataLoader(RankDatasetMock(input_sample_size[1:], config.rank),
                                              batch_size=3,
                                              num_workers=1,
                                              shuffle=False)
    return data_loader


def post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, quant_model):
    torch.cuda.set_device(config.gpu)
    quant_model.cuda(config.gpu)
    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.workers = int(config.workers / ngpus_per_node)
    quant_model = torch.nn.parallel.DistributedDataParallel(quant_model, device_ids=[config.gpu])
    compression_ctrl.distributed()
    return quant_model
