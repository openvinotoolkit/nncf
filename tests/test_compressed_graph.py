"""
 Copyright (c) 2019-2020 Intel Corporation
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
from typing import List, Tuple, Dict, Optional

import networkx as nx
import os
import pytest
import torch
import torch.nn as nn
import torchvision
from functools import partial
from copy import deepcopy

from nncf.dynamic_graph.context import get_version_agnostic_name, TracingContext
from nncf.dynamic_graph.graph import NNCFGraph, InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.graph_builder import create_input_infos, create_mock_tensor, GraphBuilder, \
    create_dummy_forward_fn, ModelInputInfo
from nncf.dynamic_graph.patch_pytorch import nncf_model_input
from nncf.layers import LSTMCellNNCF, NNCF_RNN
from nncf.model_creation import create_compression_algorithm_builders
from nncf.nncf_network import NNCFNetwork, InsertionInfo
from nncf.utils import get_all_modules_by_type
from tests import test_models
from tests.modules.seq2seq.gnmt import GNMT
from tests.modules.test_rnn import replace_lstm
from tests.helpers import get_empty_config, create_compressed_model_and_algo_for_test
from nncf.quantization.algo import PotentialQuantizedModule
from nncf.quantization.quantizer_propagation import QuantizerPropagationSolver
from nncf.quantization.layers import QuantizerConfig
from nncf.hw_config import HWConfig, HWConfigType


def get_basic_quantization_config(quantization_type, input_sample_sizes=None):
    config = get_empty_config(input_sample_sizes=input_sample_sizes)
    config["compression"] = {"algorithm": "quantization",
                             "activations": {
                                 "mode": quantization_type
                             },
                             "weights": {
                                 "mode": quantization_type
                             }}
    return config

# pylint:disable=redefined-outer-name
def get_basic_quantization_config_with_hw_config_type(hw_config_type, input_sample_size):
    config = get_empty_config(input_sample_sizes=input_sample_size)
    config["target_device"] = hw_config_type
    config["compression"] = {"algorithm": "quantization", }
    return config

def get_version_agnostic_graph(nx_graph):
    done = False
    while not done:
        counter = 0
        for node_name, node_data in nx_graph.nodes().data():
            version_specific_name = node_data["type"]
            version_agnostic_name = get_version_agnostic_name(version_specific_name)
            if version_agnostic_name != version_specific_name:
                node_data["type"] = version_agnostic_name
                mapping = dict(zip(nx_graph, nx_graph))  # identity mapping
                new_node_name = node_name.replace(version_specific_name, version_agnostic_name)
                mapping[node_name] = new_node_name
                nx_graph = nx.relabel_nodes(nx_graph, mapping, copy=False)
                break  # Looks like iterators will be invalidated after relabel_nodes
            counter += 1
        if counter == len(nx_graph.nodes().data()):
            done = True

    return nx_graph


def sort_dot(path):
    with open(path, 'r') as f:
        content = f.readlines()
    start_line = 'strict digraph  {\n'
    end_line = '}\n'
    content.remove(start_line)
    content.remove(end_line)

    def graph_key(line, offset):
        key = line.split(' ')[0].replace('"', '')
        if '->' in line:
            key += line.split(' ')[3].replace('"', '')
            return int(key) + offset
        return int(key)

    sorted_content = sorted(content, key=partial(graph_key, offset=len(content)))
    with open(path, 'w') as f:
        f.write(start_line)
        f.writelines(sorted_content)
        f.write(end_line)


def check_graph(graph: NNCFGraph, path_to_dot, graph_dir, sort_dot_graph=True):
    # pylint:disable=protected-access
    nx_graph = graph._get_graph_for_structure_analysis()

    data_dir = os.path.join(os.path.dirname(__file__), 'data/reference_graphs')
    dot_dir = os.path.join(data_dir, graph_dir)
    path_to_dot = os.path.abspath(os.path.join(dot_dir, path_to_dot))

    # validate .dot file manually!
    if not os.path.exists(path_to_dot):
        if not os.path.exists(dot_dir):
            os.makedirs(dot_dir)
        nx.drawing.nx_pydot.write_dot(nx_graph, path_to_dot)
        if sort_dot_graph:
            sort_dot(path_to_dot)

    load_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)
    load_graph = get_version_agnostic_graph(load_graph)

    # nx_graph is expected to have version-agnostic operator names already
    for k, attrs in nx_graph.nodes.items():
        attrs = {k: str(v) for k, v in attrs.items()}
        load_attrs = {k: str(v).strip('"') for k, v in load_graph.nodes[k].items()}
        if attrs != load_attrs:
            assert attrs == load_attrs

    assert load_graph.nodes.keys() == nx_graph.nodes.keys()
    assert nx.DiGraph(load_graph).edges == nx_graph.edges


class QuantizeTestCaseConfiguration:
    def __init__(self, quant_type: str, graph_dir: str):
        self.quant_type = quant_type
        self.graph_dir = graph_dir


QUANTIZERS = ['symmetric', 'asymmetric']


@pytest.fixture(scope='function', params=QUANTIZERS)
def _case_config(request):
    quantization_type = request.param
    graph_dir = os.path.join('quantized', quantization_type)
    return QuantizeTestCaseConfiguration(quantization_type, graph_dir)


def gnmt_forward_fn(seq_len, batch_size, vocab_size):
    def forward_fn(model, seq_len_, batch_size_, vocab_size_, batch_first_):
        device = next(model.parameters()).device

        def gen_packed_sequence():
            seq_list = []
            seq_lens = torch.LongTensor((batch_size_)).random_(1, seq_len_ + 1).type(torch.int32).to(device)
            seq_lens = torch.sort(seq_lens, descending=True).values
            for seq_size in seq_lens:
                seq_list.append(torch.LongTensor(seq_size.item()).random_(1, vocab_size_).to(device))
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=batch_first_)
            return padded_seq_batch, seq_lens

        x_data, seq_lens = gen_packed_sequence()
        input_encoder = x_data
        input_enc_len = seq_lens
        input_decoder = gen_packed_sequence()[0]

        nncf_model_input(input_encoder)
        nncf_model_input(input_enc_len)
        nncf_model_input(input_decoder)
        model(input_encoder, input_enc_len, input_decoder)

    return partial(forward_fn, seq_len_=seq_len, batch_size_=batch_size, vocab_size_=vocab_size, batch_first_=False)


class ModelDesc:
    def __init__(self, dot_filename: str, model_builder, input_sample_sizes, dummy_forward_fn=None):
        self.model_name = self._get_model_name(dot_filename)
        self.model_builder = model_builder
        self.dot_filename = dot_filename
        self.input_sample_sizes = input_sample_sizes

        def dummy_forward_wrapper(model_):
            return dummy_forward_fn(model_, input_sample_sizes)

        self.dummy_forward_fn = None
        if dummy_forward_fn:
            self.dummy_forward_fn = dummy_forward_wrapper

    @staticmethod
    def _get_model_name(dot_filename):
        if isinstance(dot_filename, tuple):
            dot_filename = dot_filename[0]
        return dot_filename[:dot_filename.find('.dot')]


def sr_dummy_forward_fn(model_, input_sample_sizes: Tuple[List[int]]):
    device = next(model_.parameters()).device
    config = {'input_info': [{"sample_size": sizes} for sizes in input_sample_sizes]}
    input_info_list = create_input_infos(config)
    tensor_list = [create_mock_tensor(info, device) for info in input_info_list]
    for idx, tensor in enumerate(tensor_list):
        tensor_list[idx] = nncf_model_input(tensor)
    return model_(tuple(tensor_list))


TEST_MODELS_DESC = [
    ModelDesc("alexnet.dot", test_models.AlexNet, [1, 3, 32, 32]),
    ModelDesc("lenet.dot", test_models.LeNet, [1, 3, 32, 32]),
    ModelDesc("resnet18.dot", test_models.ResNet18, [1, 3, 32, 32]),
    ModelDesc("resnet50.dot", test_models.ResNet50, [1, 3, 32, 32]),
    ModelDesc("vgg16.dot", partial(test_models.VGG, 'VGG16'), [1, 3, 32, 32]),
    ModelDesc("inception.dot", test_models.GoogLeNet, [1, 3, 32, 32]),
    ModelDesc("densenet121.dot", test_models.DenseNet121, [1, 3, 32, 32]),
    ModelDesc("inception_v3.dot", partial(test_models.Inception3, aux_logits=True, transform_input=True),
              [2, 3, 299, 299]),
    ModelDesc("squeezenet1_0.dot", test_models.squeezenet1_0, [1, 3, 32, 32]),
    ModelDesc("squeezenet1_1.dot", test_models.squeezenet1_1, [1, 3, 32, 32]),
    ModelDesc("shufflenetv2.dot", partial(test_models.ShuffleNetV2, net_size=0.5), [1, 3, 32, 32]),
    ModelDesc("shuflenet_g2.dot", test_models.ShuffleNetG2, [1, 3, 32, 32]),
    ModelDesc("ssd_vgg.dot", test_models.ssd_vgg300, [2, 3, 300, 300]),
    ModelDesc("ssd_mobilenet.dot", test_models.ssd_mobilenet, [2, 3, 300, 300]),
    ModelDesc("mobilenet_v2.dot", torchvision.models.MobileNetV2, [2, 3, 32, 32]),
    ModelDesc("resnext29_32x4d.dot", test_models.ResNeXt29_32x4d, [1, 3, 32, 32]),
    ModelDesc("pnasnetb.dot", test_models.PNASNetB, [1, 3, 32, 32]),
    ModelDesc("senet18.dot", test_models.SENet18, [1, 3, 32, 32]),
    ModelDesc("preresnet50.dot", test_models.PreActResNet50, [1, 3, 32, 32]),
    ModelDesc("unet.dot", test_models.UNet, [1, 3, 360, 480]),
    ModelDesc("lstm_cell.dot", LSTMCellNNCF, [1, 1]),
    ModelDesc("lstm_uni_seq.dot", partial(NNCF_RNN, num_layers=1, bidirectional=False), [3, 1, 1]),
    ModelDesc("lstm_uni_stacked.dot", partial(NNCF_RNN, num_layers=2, bidirectional=False), [3, 1, 1]),
    ModelDesc("lstm_bi_seq.dot", partial(NNCF_RNN, num_layers=1, bidirectional=True), [3, 1, 1]),
    ModelDesc("lstm_bi_stacked.dot", partial(NNCF_RNN, num_layers=2, bidirectional=True), [3, 1, 1]),
    ModelDesc("sr_small_model.dot", test_models.SmallModel, ([1, 3, 32, 32], [1, 3, 96, 96]), sr_dummy_forward_fn)
]


def check_model_graph(compressed_model: NNCFNetwork, ref_dot_file_name: str, ref_dot_file_directory: str):
    compressed_model.to('cuda')
    compressed_model.do_dummy_forward()
    compressed_model.do_dummy_forward()
    check_graph(compressed_model.get_graph(), ref_dot_file_name, ref_dot_file_directory)


@pytest.mark.parametrize(
    "desc", TEST_MODELS_DESC, ids=[m.model_name for m in TEST_MODELS_DESC]
)
class TestModelsGraph:
    def test_build_graph(self, desc: ModelDesc):
        net = desc.model_builder()
        input_sample_sizes = desc.input_sample_sizes
        if isinstance(input_sample_sizes, tuple):
            input_info_list = [ModelInputInfo(sample_size) for sample_size in input_sample_sizes]
        else:
            input_info_list = [ModelInputInfo(input_sample_sizes)]
        dummy_forward_fn = desc.dummy_forward_fn
        if not dummy_forward_fn:
            dummy_forward_fn = create_dummy_forward_fn(input_info_list)
        graph_builder = GraphBuilder(custom_forward_fn=dummy_forward_fn)
        graph = graph_builder.build_graph(net)
        check_graph(graph, desc.dot_filename, 'original')

    @pytest.mark.parametrize(
        "algo",
        (
            "rb_sparsity",
            "magnitude_sparsity",
            "const_sparsity",
        ), ids=['RB', 'Magnitude', 'Const']
    )
    def test_sparse_network(self, desc: ModelDesc, algo):
        # TODO: Need to fix duplicate graph for sr_small_model.
        if desc.model_name == 'sr_small_model':
            pytest.skip()
        model = desc.model_builder()
        from nncf.layers import NNCF_MODULES_MAP
        sparsifiable_modules = list(NNCF_MODULES_MAP.values())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))

        config = get_empty_config(input_sample_sizes=desc.input_sample_sizes)
        config["compression"] = {"algorithm": algo}

        compressed_model, compression_ctrl = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn)
        assert ref_num_sparsed == len(compression_ctrl.sparsified_module_info)
        check_model_graph(compressed_model, desc.dot_filename, algo)

    def test_quantize_network(self, desc: ModelDesc, _case_config):
        # TODO: Need to fix duplicate graph for sr_small_model.
        if desc.model_name == 'sr_small_model':
            pytest.skip()
        model = desc.model_builder()
        config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=desc.input_sample_sizes)
        compressed_model, _ = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn)
        check_model_graph(compressed_model, desc.dot_filename, _case_config.graph_dir)

    def test_sparse_quantize_network(self, desc: ModelDesc):
        # TODO: Need to fix duplicate graph for sr_small_model.
        if desc.model_name == 'sr_small_model':
            pytest.skip()

        model = desc.model_builder()

        from nncf.layers import NNCF_MODULES_MAP
        sparsifiable_modules = list(NNCF_MODULES_MAP.values())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))
        config = get_empty_config(input_sample_sizes=desc.input_sample_sizes)
        config["compression"] = [
            {"algorithm": "rb_sparsity"},
            {"algorithm": "quantization"}
        ]

        compressed_model, compression_ctrl = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn)

        assert ref_num_sparsed == len(compression_ctrl.child_ctrls[0].sparsified_module_info)
        check_model_graph(compressed_model, desc.dot_filename, "quantized_rb_sparsity")


def test_gnmt_quantization(_case_config):
    model = GNMT(vocab_size=32)
    model = replace_lstm(model)
    forward_fn_ = gnmt_forward_fn(seq_len=10, batch_size=3, vocab_size=32)

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=[3, 10])
    config["quantizer_setup_type"] = 'pattern_based'
    config["compression"].update({
        "quantizable_subgraph_patterns": [["linear", "__add__"],
                                          ["sigmoid", "__mul__", "__add__"],
                                          ["__add__", "tanh", "__mul__"],
                                          ["sigmoid", "__mul__"]],
        "disable_function_quantization_hooks": True,
        "ignored_scopes": ["GNMT/ResidualRecurrentEncoder[encoder]/Embedding[embedder]",
                           "GNMT/ResidualRecurrentDecoder[decoder]/Embedding[embedder]"]})

    compressed_model = NNCFNetwork(model,
                                   input_infos=create_input_infos(config),
                                   dummy_forward_fn=forward_fn_,
                                   scopes_without_shape_matching=
                                   ['GNMT/ResidualRecurrentDecoder[decoder]/RecurrentAttention[att_rnn]/'
                                    'BahdanauAttention[attn]'])

    compression_algo_builder_list = create_compression_algorithm_builders(config)

    for builder in compression_algo_builder_list:
        compressed_model = builder.apply_to(compressed_model)
    _ = compressed_model.commit_compression_changes()
    check_model_graph(compressed_model, 'gnmt_variable.dot', _case_config.graph_dir)


def test_resnet18__with_not_qinput(_case_config):
    model = test_models.ResNet18()
    input_shape = [1, 3, 32, 32]

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_shape)
    config["compression"].update({"quantize_inputs": False})

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'resnet18_no_qinput.dot', _case_config.graph_dir)


def test_resnet18__with_ignore(_case_config):
    model = test_models.ResNet18()
    input_shape = [1, 3, 32, 32]

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_shape)
    ignored_scopes = ['ResNet/Sequential[layer3]', ]
    config.update({"ignored_scopes": ignored_scopes})  # Global config ignored_scopes for NNCF module replacement
    config["compression"].update({"ignored_scopes": ignored_scopes})  # Local ignored_scopes for quantization

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'resnet18_ignore.dot', _case_config.graph_dir)


def test_iterate_module_list():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ml = nn.ModuleList([nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)])

        def forward(self, x):
            return [self.ml[0](x), self.ml[1](x)]

    net = Net()

    context = TracingContext()
    with context:
        _ = net(torch.zeros(1, 1, 1, 1))

    check_graph(context.graph, 'case_iterate_module_list.dot', 'original')


def test_output_quantization(_case_config):
    # TODO: Add support "quantize_outputs" option in propagation mode.
    pytest.skip()
    model = test_models.UNet()
    input_shape = [1, 3, 360, 480]

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_shape)
    config["compression"].update({"quantize_outputs": True})

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'unet_qoutput.dot', _case_config.graph_dir)


def test_custom_quantizable_subgraph_patterns(_case_config):
    model = test_models.SENet18()

    input_shape = [1, 3, 32, 32]

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_shape)

    config["compression"].update({"quantize_outputs": False,
                                  "quantizable_subgraph_patterns": [["sigmoid", "__mul__"],
                                                                    ["__iadd__", "batch_norm"]]})

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'senet_custom_patterns.dot', _case_config.graph_dir)

TEST_HW_MODELS_DESC = [
    ModelDesc("resnet50.dot", test_models.ResNet50, [1, 3, 32, 32]),
    ModelDesc("inception_v3.dot", partial(test_models.Inception3, aux_logits=True, transform_input=True),
              [2, 3, 299, 299]),
    ModelDesc("mobilenet_v2.dot", torchvision.models.MobileNetV2, [2, 3, 32, 32])
]

TYPE_HW = [(HWConfigType.CPU), (HWConfigType.GPU), (HWConfigType.VPU)]

@pytest.fixture(scope='function', params=TYPE_HW)
def hw_config_type(request):
    type_hw = request.param
    return type_hw

# pylint:disable=too-many-branches
@pytest.mark.parametrize(
    "desc", TEST_HW_MODELS_DESC, ids=[m.model_name for m in TEST_HW_MODELS_DESC]
)

# pylint:disable=redefined-outer-name
def test_compressed_graph_models_hw(desc, hw_config_type):
    model = desc.model_builder()
    config = get_basic_quantization_config_with_hw_config_type(hw_config_type.value,
                                                               input_sample_size=desc.input_sample_sizes)
    input_info_list = create_input_infos(config)
    hw_config_path = HWConfig.get_path_to_hw_config(hw_config_type)
    hw_config = HWConfig.from_json(hw_config_path)
    compressed_model = NNCFNetwork(model, input_infos=input_info_list)

    # pylint:disable=protected-access
    compression_algo_builder = create_compression_algorithm_builders(config)[0]
    potential_weights_modules =\
        compression_algo_builder.get_potential_quantized_modules(compressed_model)
    prop_graph_solver = QuantizerPropagationSolver(hw_config=hw_config)
    insertion_point_graph = compressed_model.get_insertion_point_graph()
    merged_ip_graph = insertion_point_graph.get_ip_graph_with_merged_hw_optimized_operations(hw_config)
    potential_activations_quantizers = prop_graph_solver.run_on_ip_graph(merged_ip_graph)
    sketch_graph = compressed_model.get_original_graph()

    potential_quantizer_graph = prepare_potential_quantizer_graph(sketch_graph, potential_activations_quantizers,
                                                                  potential_weights_modules)
    check_graph(potential_quantizer_graph, desc.dot_filename, _case_dir(hw_config_type.value), sort_dot_graph=False)

def _case_dir(type_hw_config):
    graph_dir = os.path.join('quantized', "hw", type_hw_config)
    return graph_dir

def prepare_potential_quantizer_graph(graph: NNCFGraph,
                                      potential_activations_quantizers: Dict[InsertionInfo,
                                                                             Optional[List[QuantizerConfig]]],
                                      potential_weights_modules: List[PotentialQuantizedModule]) -> NNCFGraph:
    quantizers_weights_attr = {}
    quantizers_activations_attr = {}
    # pylint:disable=protected-access
    for _, module_scope, qconfig_list in potential_weights_modules:
        matching_graph_op_nodes = graph.get_op_nodes_in_scope(module_scope)

        assert len(matching_graph_op_nodes) == 1  # Isn't correct when NNCF module has more than 1 graph node

        op_name = matching_graph_op_nodes[0][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
        ia_op_exec_context = InputAgnosticOperationExecutionContext(op_name, module_scope, 0)
        str_qconfig_list = ''

        for qconfig in qconfig_list:
            str_qconfig_list += '[' + str(qconfig) + '] '
        quantizers_weights_attr[ia_op_exec_context] = str_qconfig_list

    for insertion_info, qconfig_list in potential_activations_quantizers.items():
        ia_op_exec_context = insertion_info.op_exec_context.input_agnostic
        str_qconfig_list = ''
        for qconfig in qconfig_list:
            str_qconfig_list += '[' + str(qconfig) + '] '
        quantizers_activations_attr[ia_op_exec_context] = str_qconfig_list
        for linked_op_exec_context in insertion_info.linked_op_exec_contexts:
            quantizers_activations_attr[linked_op_exec_context.input_agnostic] = str_qconfig_list

    nx_graph = graph._nx_graph
    nodes = deepcopy(nx_graph.nodes)
    for node_name, node in sorted(nodes.items()):
        ia_op_exec_context_for_node = nx_graph.nodes[node_name][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic
        scope_node = str(ia_op_exec_context_for_node)
        if ia_op_exec_context_for_node in quantizers_activations_attr:
            label = "Quantizer: {}".format(quantizers_activations_attr[ia_op_exec_context_for_node])
            nx_graph.add_node(scope_node, label=label, color="purple", id=node[NNCFGraph.ID_NODE_ATTR],
                              op_exec_context=nx_graph.nodes[node_name][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR])
            next_nodes = deepcopy(nx_graph._succ[node_name])
            for next_node_name, _ in next_nodes.items():
                nx_graph.add_edge(scope_node, next_node_name)
                nx_graph.remove_edge(node_name, next_node_name)
            nx_graph.add_edge(node_name, scope_node)
        elif ia_op_exec_context_for_node in quantizers_weights_attr:
            label = "Quantizer: {}".format(quantizers_weights_attr[ia_op_exec_context_for_node])
            nx_graph.add_node(scope_node, label=label, color="purple", id=node[NNCFGraph.ID_NODE_ATTR],
                              op_exec_context=nx_graph.nodes[node_name][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR])
            nx_graph.add_edge(scope_node, node_name)

    return graph
