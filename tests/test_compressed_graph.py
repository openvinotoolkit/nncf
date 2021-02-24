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
from abc import abstractmethod, ABC
from typing import List, Tuple, Dict, Callable, Union

import networkx as nx
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from functools import partial
from copy import deepcopy

from nncf.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.dynamic_graph.context import get_version_agnostic_name, TracingContext
from nncf.dynamic_graph.graph import NNCFGraph, InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.graph_builder import create_input_infos, create_mock_tensor, GraphBuilder, \
    create_dummy_forward_fn, ModelInputInfo
from nncf import nncf_model_input
from nncf.layers import LSTMCellNNCF, NNCF_RNN
from nncf.nncf_network import NNCFNetwork, InsertionType
from nncf.utils import get_all_modules_by_type
from tests import test_models
from tests.modules.seq2seq.gnmt import GNMT
from tests.modules.test_rnn import replace_lstm
from tests.helpers import get_empty_config, create_compressed_model_and_algo_for_test
from nncf.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.hw_config import HWConfigType
from tests.test_models.synthetic import ManyNonEvalModules, PoolUnPool, ArangeModel, TransposeModel, \
    GatherModel, MaskedFillModel, ReshapeModel, ModelWithDummyParameter


def get_basic_quantization_config(quantization_type='symmetric', input_sample_sizes=None, input_info=None):
    config = get_empty_config(input_sample_sizes=input_sample_sizes, input_info=input_info)
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


def gnmt_wrap_inputs_fn(model_args, model_kwargs):
    # Assuming 3 args to wrap: input_encoder, input_enc_len, input_decoder, and 0 kwargs to wrap
    model_args = (nncf_model_input(model_args[0]),
                  nncf_model_input(model_args[1]),
                  nncf_model_input(model_args[2]))
    return model_args, model_kwargs


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

        args, kwargs = gnmt_wrap_inputs_fn((input_encoder, input_enc_len, input_decoder), {})
        model(*args, **kwargs)

    return partial(forward_fn, seq_len_=seq_len, batch_size_=batch_size, vocab_size_=vocab_size, batch_first_=False)


class ModelDesc:
    def __init__(self, model_name: str, model_builder, input_sample_sizes, dummy_forward_fn=None,
                 wrap_inputs_fn=None):
        self.model_name = model_name
        self.model_builder = model_builder
        self.input_sample_sizes = input_sample_sizes

        def dummy_forward_wrapper(model_):
            return dummy_forward_fn(model_, input_sample_sizes)

        self.dummy_forward_fn = None
        if dummy_forward_fn:
            self.dummy_forward_fn = dummy_forward_wrapper

        self.wrap_inputs_fn = wrap_inputs_fn

    @property
    def dot_filename(self):
        return self.model_name + '.dot'


def sr_wrap_inputs_fn(model_args, model_kwargs):
    # Assuming 2 tensors in the 0-th arg (tuple) to wrap and 0 kwargs to wrap
    model_args = ((nncf_model_input(model_args[0][0]),
                   nncf_model_input(model_args[0][1])),)
    return model_args, model_kwargs


def sr_dummy_forward_fn(model_, input_sample_sizes: Tuple[List[int]]):
    device = next(model_.parameters()).device
    config = {'input_info': [{"sample_size": sizes} for sizes in input_sample_sizes]}
    input_info_list = create_input_infos(config)
    tensor_list = [create_mock_tensor(info, device) for info in input_info_list]
    args = (tuple(tensor_list),)
    args, _ = sr_wrap_inputs_fn(args, {})
    return model_(*args)


TEST_MODELS_DESC = [
    ModelDesc("alexnet", test_models.AlexNet, [1, 3, 32, 32]),
    ModelDesc("lenet", test_models.LeNet, [1, 3, 32, 32]),
    ModelDesc("resnet18", test_models.ResNet18, [1, 3, 32, 32]),
    ModelDesc("resnet50", test_models.ResNet50, [1, 3, 32, 32]),
    ModelDesc("vgg16", partial(test_models.VGG, 'VGG16'), [1, 3, 32, 32]),
    ModelDesc("inception", test_models.GoogLeNet, [1, 3, 32, 32]),
    ModelDesc("densenet121", test_models.DenseNet121, [1, 3, 32, 32]),
    ModelDesc("inception_v3", partial(test_models.Inception3, aux_logits=True, transform_input=True),
              [2, 3, 299, 299]),
    ModelDesc("squeezenet1_0", test_models.squeezenet1_0, [1, 3, 32, 32]),
    ModelDesc("squeezenet1_1", test_models.squeezenet1_1, [1, 3, 32, 32]),
    ModelDesc("shufflenetv2", partial(test_models.ShuffleNetV2, net_size=0.5), [1, 3, 32, 32]),
    ModelDesc("shuflenet_g2", test_models.ShuffleNetG2, [1, 3, 32, 32]),
    ModelDesc("ssd_vgg", test_models.ssd_vgg300, [2, 3, 300, 300]),
    ModelDesc("ssd_mobilenet", test_models.ssd_mobilenet, [2, 3, 300, 300]),
    ModelDesc("mobilenet_v2", torchvision.models.MobileNetV2, [2, 3, 32, 32]),
    ModelDesc("resnext29_32x4d", test_models.ResNeXt29_32x4d, [1, 3, 32, 32]),
    ModelDesc("pnasnetb", test_models.PNASNetB, [1, 3, 32, 32]),
    ModelDesc("senet18", test_models.SENet18, [1, 3, 32, 32]),
    ModelDesc("preresnet50", test_models.PreActResNet50, [1, 3, 32, 32]),
    ModelDesc("unet", test_models.UNet, [1, 3, 360, 480]),
    ModelDesc("lstm_cell", LSTMCellNNCF, [1, 1]),
    ModelDesc("lstm_uni_seq", partial(NNCF_RNN, num_layers=1, bidirectional=False), [3, 1, 1]),
    ModelDesc("lstm_uni_stacked", partial(NNCF_RNN, num_layers=2, bidirectional=False), [3, 1, 1]),
    ModelDesc("lstm_bi_seq", partial(NNCF_RNN, num_layers=1, bidirectional=True), [3, 1, 1]),
    ModelDesc("lstm_bi_stacked", partial(NNCF_RNN, num_layers=2, bidirectional=True), [3, 1, 1]),
    ModelDesc("sr_small_model", test_models.SmallModel, ([1, 3, 32, 32], [1, 3, 96, 96]),
              dummy_forward_fn=sr_dummy_forward_fn,
              wrap_inputs_fn=sr_wrap_inputs_fn)
]


def check_model_graph(compressed_model: NNCFNetwork, ref_dot_file_name: str, ref_dot_file_directory: str):
    compressed_model.to('cuda')
    compressed_model.do_dummy_forward()
    # internal wrapped model is still in eval mode, switch to the train mode to make sure training graph is ok
    compressed_model.train()
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
            dummy_forward_fn = create_dummy_forward_fn(input_info_list, desc.wrap_inputs_fn)
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
        model = desc.model_builder()
        from nncf.layers import NNCF_MODULES_MAP

        config = get_empty_config(input_sample_sizes=desc.input_sample_sizes)
        config["compression"] = {"algorithm": algo}

        compressed_model, compression_ctrl = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn,
                                                      wrap_inputs_fn=desc.wrap_inputs_fn)

        # counts wrapped NNCF modules to ignore the ones that are called in the training mode only
        sparsifiable_modules = list(NNCF_MODULES_MAP.keys())
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))
        assert ref_num_sparsed == len(compression_ctrl.sparsified_module_info)
        check_model_graph(compressed_model, desc.dot_filename, algo)

    def test_quantize_network(self, desc: ModelDesc, _case_config):
        model = desc.model_builder()
        config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=desc.input_sample_sizes)
        compressed_model, _ = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn,
                                                      wrap_inputs_fn=desc.wrap_inputs_fn)
        check_model_graph(compressed_model, desc.dot_filename, _case_config.graph_dir)

    def test_sparse_quantize_network(self, desc: ModelDesc):
        model = desc.model_builder()

        from nncf.layers import NNCF_MODULES_MAP
        config = get_empty_config(input_sample_sizes=desc.input_sample_sizes)
        config["compression"] = [
            {"algorithm": "rb_sparsity"},
            {"algorithm": "quantization"}
        ]

        compressed_model, compression_ctrl = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn,
                                                      wrap_inputs_fn=desc.wrap_inputs_fn)

        # counts wrapped NNCF modules to ignore the ones that are called in the training mode only
        sparsifiable_modules = list(NNCF_MODULES_MAP.keys())
        ref_num_sparsed = len(get_all_modules_by_type(compressed_model, sparsifiable_modules))

        assert ref_num_sparsed == len(compression_ctrl.child_ctrls[0].sparsified_module_info)
        check_model_graph(compressed_model, desc.dot_filename, "quantized_rb_sparsity")


def test_gnmt_quantization(_case_config):
    model = GNMT(vocab_size=32)
    model = replace_lstm(model)
    forward_fn_ = gnmt_forward_fn(seq_len=10, batch_size=3, vocab_size=32)

    config = get_basic_quantization_config(_case_config.quant_type)
    config["input_info"] = [
        {
            "sample_size": [3, 10],
            "type": "long"
        },
        {
            "sample_size": [3],
            "type": "long"
        },
        {
            "sample_size": [3, 10],
            "type": "long"
        }
    ]
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
                                   wrap_inputs_fn=gnmt_wrap_inputs_fn,
                                   scopes_without_shape_matching=
                                   ['GNMT/ResidualRecurrentDecoder[decoder]/RecurrentAttention[att_rnn]/'
                                    'BahdanauAttention[attn]'])

    composite_builder = PTCompositeCompressionAlgorithmBuilder(config)
    composite_builder.apply_to(compressed_model)

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


def n_inputs_fn(model_args, model_kwargs, nargs=2):
    model_args = tuple(nncf_model_input(model_args[i]) for i in range(nargs))
    return model_args, model_kwargs


def cat_two_inputs_fn(model_args, model_kwargs):
    model_args = tuple(nncf_model_input(model_args[i]) for i in range(2))
    return (model_args,), model_kwargs


class IModelDesc(ABC):

    @abstractmethod
    def get_input_sample_sizes(self):
        pass

    @abstractmethod
    def get_input_info(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_dot_filename(self):
        pass

    @abstractmethod
    def get_wrap_inputs_fn(self):
        pass


class BaseDesc(IModelDesc):
    def __init__(self, input_sample_sizes: Union[Tuple[List[int], ...], List[int]] = None,
                 model_name: str = None, wrap_inputs_fn: Callable = None, input_info=None,
                 dummy_forward_fn=None):
        self.input_sample_sizes = input_sample_sizes
        self.input_info = input_info
        self.model_name = model_name
        self.wrap_inputs_fn = wrap_inputs_fn
        self.dummy_forward_fn = dummy_forward_fn

    def get_input_sample_sizes(self):
        return self.input_sample_sizes

    def get_input_info(self):
        return self.input_info

    def get_dot_filename(self):
        return self.model_name + '.dot'

    def get_wrap_inputs_fn(self):
        return self.wrap_inputs_fn

    def get_dummy_forward_fn(self):
        return self.dummy_forward_fn

    @abstractmethod
    def get_model(self):
        pass


class GeneralModelDesc(BaseDesc):
    def __init__(self,
                 input_sample_sizes: Union[Tuple[List[int], ...], List[int]] = None,
                 model_name: str = None, wrap_inputs_fn: Callable = None, model_builder=None, input_info=None):
        super().__init__(input_sample_sizes, model_name, wrap_inputs_fn, input_info)
        if not model_name and hasattr(model_builder, '__name__'):
            self.model_name = model_builder.__name__
        self.model_builder = model_builder

    def get_model(self):
        return self.model_builder()


class SingleLayerModelDesc(BaseDesc):
    def __init__(self, layer: nn.Module,
                 input_sample_sizes: Union[Tuple[List[int], ...], List[int]] = None, model_name: str = None,
                 wrap_inputs_fn: Callable = None, input_info=None):
        super().__init__(input_sample_sizes, model_name, wrap_inputs_fn, input_info)

        self.model_name = model_name
        if model_name is None:
            self.model_name = layer.__class__.__name__

        self.layer = layer
        self.input_sample_sizes = input_sample_sizes if input_sample_sizes else [1]
        self.wrap_inputs_fn = wrap_inputs_fn
        if wrap_inputs_fn is None:
            self.wrap_inputs_fn = partial(n_inputs_fn, nargs=1)

    def get_model(self):
        class TestModel(ModelWithDummyParameter):
            def __init__(self, layer):
                super().__init__()
                self._layer = layer

            def forward(self, *args, **kwargs):
                return self._layer(*args, **kwargs)

        return TestModel(self.layer)


class TorchBinaryMethodDesc(SingleLayerModelDesc):
    def __init__(self, model_name: str, torch_method: Callable, input_info=None):
        super().__init__(layer=torch_method, model_name=model_name, input_sample_sizes=([1], [1]),
                         wrap_inputs_fn=n_inputs_fn, input_info=input_info)


class TensorBinaryMethodsDesc(BaseDesc):
    def __init__(self, tensor_method: str, model_name: str = None, input_info=None):
        super().__init__(input_sample_sizes=([1], [1]), wrap_inputs_fn=n_inputs_fn, model_name=model_name,
                         input_info=input_info)

        self.model_name = model_name
        if model_name is None:
            self.model_name = tensor_method
        self.tensor_method = tensor_method

    def get_model(self):
        class TestModel(ModelWithDummyParameter):
            def __init__(self, tensor_method):
                super().__init__()
                self._tensor_method = tensor_method

            def forward(self, t1, t2):
                return getattr(t1, self._tensor_method)(t2)

        return TestModel(self.tensor_method)


class TensorUnaryMethodsDesc(BaseDesc):
    def __init__(self, tensor_method: str, model_name: str = None, input_info=None, **model_kwargs):
        super().__init__(input_sample_sizes=([1]), wrap_inputs_fn=partial(n_inputs_fn, nargs=1), model_name=model_name,
                         input_info=input_info)
        self.model_name = model_name
        if model_name is None:
            self.model_name = tensor_method
        self.tensor_method = tensor_method
        self.model_kwargs = model_kwargs

    def get_model(self):
        class TestModel(ModelWithDummyParameter):
            def __init__(self, tensor_method, **model_kwargs):
                super().__init__()
                self._tensor_method = tensor_method
                self.model_kwargs = model_kwargs

            def forward(self, x):
                if self.model_kwargs:
                    return getattr(x, self._tensor_method)(**self.model_kwargs)
                return getattr(x, self._tensor_method)()

        return TestModel(self.tensor_method, **self.model_kwargs)


TWO_INT_INPUTS_INFO = [{"sample_size": [1], "type": "long"}, {"sample_size": [1], "type": "long"}]
SYNTHETIC_MODEL_DESC_LIST = [
    SingleLayerModelDesc(layer=nn.Conv1d(1, 1, 1), input_sample_sizes=[1, 1, 1]),
    SingleLayerModelDesc(layer=nn.Conv2d(1, 1, 1), input_sample_sizes=[1, 1, 1, 1]),
    SingleLayerModelDesc(layer=nn.ConvTranspose2d(1, 1, 1), input_sample_sizes=[1, 1, 1, 1]),
    SingleLayerModelDesc(layer=nn.Conv3d(1, 1, 1), input_sample_sizes=[1, 1, 1, 1, 1]),
    SingleLayerModelDesc(layer=nn.ConvTranspose3d(1, 1, 1), input_sample_sizes=[1, 1, 1, 1, 1]),

    SingleLayerModelDesc(layer=nn.Linear(1, 1)),

    SingleLayerModelDesc(layer=nn.Embedding(1, 1), model_name='Embedding_module',
                         input_info={"sample_size": [1, 1], "type": "long", "filler": "zeros"}),
    SingleLayerModelDesc(layer=nn.EmbeddingBag(1, 1),
                         input_info={"sample_size": [1, 1], "type": "long", "filler": "zeros"}),

    SingleLayerModelDesc(layer=nn.Hardtanh()),
    SingleLayerModelDesc(layer=nn.Tanh()),
    SingleLayerModelDesc(layer=nn.ELU()),
    SingleLayerModelDesc(layer=nn.PReLU()),
    SingleLayerModelDesc(layer=nn.LeakyReLU()),
    SingleLayerModelDesc(layer=nn.LayerNorm(normalized_shape=[1])),
    SingleLayerModelDesc(layer=nn.GELU()),
    SingleLayerModelDesc(layer=nn.Sigmoid()),

    TorchBinaryMethodDesc('Add', torch.add),
    TensorBinaryMethodsDesc('__add__'),
    TensorBinaryMethodsDesc('__radd__'),
    TensorBinaryMethodsDesc('__iadd__'),

    TorchBinaryMethodDesc('Sub', torch.sub),
    TensorBinaryMethodsDesc('__sub__'),
    TensorBinaryMethodsDesc('__rsub__'),
    TensorBinaryMethodsDesc('__isub__'),

    TorchBinaryMethodDesc('torch_mul', torch.mul),
    TensorBinaryMethodsDesc('mul', model_name='tensor_mul'),
    TensorBinaryMethodsDesc('__mul__'),
    TensorBinaryMethodsDesc('__rmul__'),
    TensorBinaryMethodsDesc('__imul__'),

    TorchBinaryMethodDesc('Div', torch.div),
    TensorBinaryMethodsDesc('__div__'),
    TensorBinaryMethodsDesc('__idiv__'),
    TensorBinaryMethodsDesc('__truediv__'),

    SingleLayerModelDesc(model_name='Exp', layer=torch.exp),
    SingleLayerModelDesc(model_name='Erf', layer=torch.erf),

    TorchBinaryMethodDesc(model_name='MatMul', torch_method=torch.matmul),
    SingleLayerModelDesc(model_name='BMM', layer=torch.bmm, input_sample_sizes=([1, 1, 1], [1, 1, 1]),
                         wrap_inputs_fn=n_inputs_fn),
    TensorBinaryMethodsDesc(model_name='MatMul', tensor_method='matmul'),

    SingleLayerModelDesc(model_name='Mean', layer=torch.mean),

    TensorUnaryMethodsDesc(tensor_method='round'),

    SingleLayerModelDesc(layer=nn.Dropout()),
    SingleLayerModelDesc(layer=nn.Threshold(0.1, 20)),

    SingleLayerModelDesc(layer=nn.BatchNorm1d(1), input_sample_sizes=([2, 1, 1])),
    SingleLayerModelDesc(layer=nn.BatchNorm2d(1), input_sample_sizes=([2, 1, 1, 1])),
    SingleLayerModelDesc(layer=nn.BatchNorm3d(1), input_sample_sizes=([2, 1, 1, 1, 1])),

    SingleLayerModelDesc(layer=nn.AvgPool2d(1), input_sample_sizes=[1, 1, 1]),
    SingleLayerModelDesc(layer=nn.AdaptiveAvgPool2d(1), input_sample_sizes=[1, 1, 1]),
    SingleLayerModelDesc(layer=nn.AvgPool3d(1), input_sample_sizes=[1, 1, 1, 1]),
    SingleLayerModelDesc(layer=nn.AdaptiveAvgPool3d(1), input_sample_sizes=[1, 1, 1, 1]),

    SingleLayerModelDesc(layer=nn.MaxPool1d(1), input_sample_sizes=[1, 1, 1]),
    SingleLayerModelDesc(layer=nn.MaxPool2d(1), input_sample_sizes=[1, 1, 1]),
    SingleLayerModelDesc(layer=nn.MaxPool3d(1), input_sample_sizes=[1, 1, 1, 1]),

    GeneralModelDesc(model_name='MaxUnpool3d', model_builder=PoolUnPool,
                     input_info={"sample_size": [1, 1, 3, 3, 3], "type": "float", "filler": "random"}),

    SingleLayerModelDesc(model_name='pad', layer=partial(F.pad, pad=[1, 1]), input_sample_sizes=([2, 2])),
    SingleLayerModelDesc(model_name='cat', layer=partial(torch.cat, dim=0), wrap_inputs_fn=cat_two_inputs_fn,
                         input_sample_sizes=([1], [1])),
    SingleLayerModelDesc(model_name='stack', layer=partial(torch.stack, dim=0), wrap_inputs_fn=cat_two_inputs_fn,
                         input_sample_sizes=([1], [1])),

    SingleLayerModelDesc(model_name='relu', layer=torch.relu),
    SingleLayerModelDesc(model_name='relu_', layer=torch.relu_),

    SingleLayerModelDesc(model_name='max', layer=torch.max),
    SingleLayerModelDesc(model_name='min', layer=torch.min),

    GeneralModelDesc(model_builder=ArangeModel),

    SingleLayerModelDesc(model_name='transpose', layer=partial(torch.transpose, dim0=0, dim1=0)),
    GeneralModelDesc(model_builder=TransposeModel, input_sample_sizes=([1])),

    GeneralModelDesc(model_builder=GatherModel),

    GeneralModelDesc(model_builder=MaskedFillModel),
    GeneralModelDesc(model_builder=ReshapeModel, input_sample_sizes=([1])),

    TensorUnaryMethodsDesc(tensor_method='contiguous'),
    TensorUnaryMethodsDesc(tensor_method='split', split_size=(1,)),
    TensorUnaryMethodsDesc(tensor_method='chunk', chunks=1),
    TensorUnaryMethodsDesc(tensor_method='expand', size=(1,)),

    TorchBinaryMethodDesc(model_name='embedding_function', torch_method=F.embedding,
                          input_info=[{"sample_size": [1], "type": "long"}, {"sample_size": [2]}]),
    SingleLayerModelDesc(model_name='embedding_bag', layer=F.embedding_bag,
                         wrap_inputs_fn=partial(n_inputs_fn, nargs=3),
                         input_info=[{"sample_size": [1, 1]},
                                     {"sample_size": [1], "type": "long", "filler":"zeros"},
                                     {"sample_size": [1], "type": "long", "filler":"zeros"}]),

    SingleLayerModelDesc(model_name='softmax', layer=F.softmax),

    TensorBinaryMethodsDesc(tensor_method='__lt__'),
    TensorBinaryMethodsDesc(tensor_method='__le__'),
    TensorBinaryMethodsDesc(tensor_method='__gt__'),
    TensorBinaryMethodsDesc(tensor_method='__mod__'),
    TensorBinaryMethodsDesc(tensor_method='__eq__'),
    TensorBinaryMethodsDesc(tensor_method='__ne__'),
    TensorBinaryMethodsDesc(tensor_method='__or__', input_info=TWO_INT_INPUTS_INFO),
    TensorBinaryMethodsDesc(tensor_method='__xor__', input_info=TWO_INT_INPUTS_INFO),
    TensorBinaryMethodsDesc(tensor_method='__and__', input_info=TWO_INT_INPUTS_INFO),
    TensorUnaryMethodsDesc(tensor_method='logical_not_'),
    TensorBinaryMethodsDesc(tensor_method='__pow__'),
    SingleLayerModelDesc(model_name='interpolate', layer=partial(F.interpolate, size=1),
                         input_sample_sizes=([1, 1, 1])),

    SingleLayerModelDesc(model_name='repeat_interleave', layer=partial(torch.repeat_interleave, repeats=2)),
    TensorUnaryMethodsDesc(tensor_method='clone'),

    SingleLayerModelDesc(model_name='pixel_shuffle', layer=partial(F.pixel_shuffle, upscale_factor=1),
                         input_sample_sizes=([1, 1, 1, 1])),

    GeneralModelDesc(model_builder=ManyNonEvalModules, input_sample_sizes=([1, 1, 1, 1]))
]


@pytest.mark.parametrize(
    "synthetic_model_desc", SYNTHETIC_MODEL_DESC_LIST, ids=[m.model_name for m in SYNTHETIC_MODEL_DESC_LIST]
)
def test_synthetic_model_quantization(synthetic_model_desc: IModelDesc):
    config = get_basic_quantization_config(input_sample_sizes=synthetic_model_desc.get_input_sample_sizes(),
                                           input_info=synthetic_model_desc.get_input_info())

    model = synthetic_model_desc.get_model()
    compressed_model, _ = create_compressed_model_and_algo_for_test(
        model, config, wrap_inputs_fn=synthetic_model_desc.get_wrap_inputs_fn())

    check_model_graph(compressed_model, synthetic_model_desc.get_dot_filename(),
                      os.path.join('quantized', 'synthetic_model'))


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
    ModelDesc("resnet50", test_models.ResNet50, [1, 3, 32, 32]),
    ModelDesc("inception_v3", partial(test_models.Inception3, aux_logits=True, transform_input=True),
              [2, 3, 299, 299]),
    ModelDesc("mobilenet_v2", torchvision.models.MobileNetV2, [2, 3, 32, 32])
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
    compressed_model = NNCFNetwork(model, input_infos=input_info_list)

    # pylint:disable=protected-access
    quantization_builder = PTCompositeCompressionAlgorithmBuilder(config).child_builders[0]  # type: QuantizationBuilder
    single_config_quantizer_setup = quantization_builder._get_quantizer_setup(compressed_model)
    sketch_graph = compressed_model.get_original_graph()

    potential_quantizer_graph = prepare_potential_quantizer_graph(sketch_graph, single_config_quantizer_setup)
    check_graph(potential_quantizer_graph, desc.dot_filename, _case_dir(hw_config_type.value), sort_dot_graph=False)


def _case_dir(type_hw_config):
    graph_dir = os.path.join('quantized', "hw", type_hw_config)
    return graph_dir


def prepare_potential_quantizer_graph(graph: NNCFGraph,
                                      quantizer_setup: SingleConfigQuantizerSetup) -> NNCFGraph:
    quantizers_weights_attr = {}
    pre_hooked_quantizers_activations_attr = {}  # type: Dict[InputAgnosticOperationExecutionContext, Tuple[int, str]]
    post_hooked_quantizers_activations_attr = {}  # type: Dict[InputAgnosticOperationExecutionContext, str]

    # pylint:disable=protected-access
    for qp in quantizer_setup.quantization_points.values():
        if qp.is_weight_quantization_point():
            module_scope = qp.insertion_point.module_scope
            qconfig = qp.qconfig
            matching_graph_op_nodes = graph.get_op_nodes_in_scope(module_scope)

            assert len(matching_graph_op_nodes) == 1  # Isn't correct when NNCF module has more than 1 graph node

            op_name = matching_graph_op_nodes[0][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].operator_name
            ia_op_exec_context = InputAgnosticOperationExecutionContext(op_name, module_scope, 0)
            str_qconfig = str(qconfig)
            quantizers_weights_attr[ia_op_exec_context] = str_qconfig
        elif qp.is_activation_quantization_point():
            ia_op_exec_context = qp.insertion_point.ia_op_exec_context
            qconfig = qp.qconfig
            str_qconfig = str(qconfig)
            if qp.insertion_point.insertion_type is InsertionType.OPERATOR_PRE_HOOK:
                pre_hooked_quantizers_activations_attr[ia_op_exec_context] = \
                    (qp.insertion_point.input_port_id, str_qconfig)
            elif qp.insertion_point.insertion_type is InsertionType.OPERATOR_POST_HOOK:
                post_hooked_quantizers_activations_attr[ia_op_exec_context] = str_qconfig

    nx_graph = graph._nx_graph
    nodes = deepcopy(nx_graph.nodes)
    for node_name, node in sorted(nodes.items()):
        ia_op_exec_context_for_node = nx_graph.nodes[node_name][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic
        node_scope = str(ia_op_exec_context_for_node)
        if ia_op_exec_context_for_node in pre_hooked_quantizers_activations_attr:
            in_port_id, qconf_str = pre_hooked_quantizers_activations_attr[ia_op_exec_context_for_node]
            label = "Quantizer: {}".format(qconf_str)
            additional_node_attrs = dict(label=label, color="purple", id=node[NNCFGraph.ID_NODE_ATTR],
                                         op_exec_context=nx_graph.nodes[node_name][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR])

            node_scope_for_input = node_scope + '|IN' + str(in_port_id)
            nx_graph.add_node(node_scope_for_input, **additional_node_attrs)
            # Adding a pre-hook quantizer to a corresponding input port
            edges_with_matching_in_port_id = []

            for from_key, to_key, edge_attrs in nx_graph.in_edges(node_name, data=True):
                if edge_attrs[NNCFGraph.IN_PORT_NAME_EDGE_ATTR] == in_port_id:
                    edges_with_matching_in_port_id.append((from_key, to_key))

            assert len(edges_with_matching_in_port_id) == 1
            input_edge_to_break = edges_with_matching_in_port_id[0]

            existing_edge_attrs = deepcopy(nx_graph.edges[input_edge_to_break])
            nx_graph.remove_edge(input_edge_to_break[0], input_edge_to_break[1])
            nx_graph.add_edge(input_edge_to_break[0], node_scope_for_input)
            nx_graph.add_edge(node_scope_for_input, input_edge_to_break[1], **existing_edge_attrs)

        if ia_op_exec_context_for_node in post_hooked_quantizers_activations_attr:
            qconf_str = post_hooked_quantizers_activations_attr[ia_op_exec_context_for_node]
            label = "Quantizer: {}".format(qconf_str)
            additional_node_attrs = dict(label=label, color="purple", id=node[NNCFGraph.ID_NODE_ATTR],
                                         op_exec_context=nx_graph.nodes[node_name][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR])

            # Adding a post-hook quantizer for the op
            nx_graph.add_node(node_scope, **additional_node_attrs)
            next_nodes = deepcopy(nx_graph._succ[node_name])
            for next_node_name, next_node_attrs in next_nodes.items():
                existing_edge_attrs = deepcopy(next_node_attrs)
                nx_graph.add_edge(node_scope, next_node_name, **existing_edge_attrs)
                nx_graph.remove_edge(node_name, next_node_name)
            nx_graph.add_edge(node_name, node_scope)

        if ia_op_exec_context_for_node in quantizers_weights_attr:
            label = "Quantizer: {}".format(quantizers_weights_attr[ia_op_exec_context_for_node])
            nx_graph.add_node(node_scope, label=label, color="purple", id=node[NNCFGraph.ID_NODE_ATTR],
                              op_exec_context=nx_graph.nodes[node_name][NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR])
            nx_graph.add_edge(node_scope, node_name)

    return graph
