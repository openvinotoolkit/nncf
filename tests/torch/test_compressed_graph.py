"""
 Copyright (c) 2019-2022 Intel Corporation
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

import os
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import networkx as nx
import pytest
import torch

from nncf.torch.utils import get_model_device
from tests.torch.test_models.synthetic import ConvBNLeakyReLU
from tests.torch.test_models.synthetic import ConvGeluGetItem
from tests.torch.test_models.synthetic import ConvRelu6HSwishHSigmoid
from tests.torch.test_models.synthetic import FC_ConstMul
from tests.torch.test_models.synthetic import MMDivConv
from tests.torch.test_models.synthetic import MatMulDivConv
from torch import nn
import torch.nn.functional as F
import torchvision

from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.torch import nncf_model_input
from nncf.torch import nncf_model_output
from nncf.common.graph import NNCFNodeName
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos
from nncf.torch.dynamic_graph.graph_tracer import create_mock_tensor
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.layers import LSTMCellNNCF
from nncf.torch.layers import NNCF_RNN
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.algo import QuantizationBuilder
from nncf.torch.utils import get_all_modules_by_type
from tests.torch import test_models
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.modules.seq2seq.gnmt import GNMT
from tests.torch.modules.test_rnn import replace_lstm
from nncf.torch.layers import NNCF_MODULES_DICT
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from tests.torch.test_models.synthetic import ArangeModel
from tests.torch.test_models.synthetic import EmbeddingCatLinearModel
from tests.torch.test_models.synthetic import EmbeddingSumModel
from tests.torch.test_models.synthetic import GatherModel
from tests.torch.test_models.synthetic import ManyNonEvalModules
from tests.torch.test_models.synthetic import MaskedFillModel
from tests.torch.test_models.synthetic import ModelWithDummyParameter
from tests.torch.test_models.synthetic import MultiOutputSameTensorModel
from tests.torch.test_models.synthetic import PoolUnPool
from tests.torch.test_models.synthetic import ReshapeModel
from tests.torch.test_models.synthetic import TransposeModel


def get_basic_quantization_config(quantization_type='symmetric', input_sample_sizes=None, input_info: Dict = None):
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


def sort_dot(path):
    with open(path, 'r', encoding='utf8') as f:
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
    with open(path, 'w', encoding='utf8') as f:
        f.write(start_line)
        f.writelines(sorted_content)
        f.write(end_line)


def check_graph(graph: PTNNCFGraph, path_to_dot, graph_dir, sort_dot_graph=True):
    # pylint:disable=protected-access
    nx_graph = graph.get_graph_for_structure_analysis()
    check_nx_graph(nx_graph, path_to_dot, graph_dir, sort_dot_graph=sort_dot_graph)


def check_nx_graph(nx_graph: nx.DiGraph, path_to_dot, graph_dir, sort_dot_graph=True):
    data_dir = os.path.join(os.path.dirname(__file__), 'data/reference_graphs')
    dot_dir = os.path.join(data_dir, graph_dir)
    path_to_dot = os.path.abspath(os.path.join(dot_dir, path_to_dot))

    for _, node in nx_graph.nodes(data=True):
        if 'scope' in node:
            node.pop('scope')

    # validate .dot file manually!
    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        if not os.path.exists(dot_dir):
            os.makedirs(dot_dir)
        nx.drawing.nx_pydot.write_dot(nx_graph, path_to_dot)
        if sort_dot_graph:
            sort_dot(path_to_dot)

    load_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)

    # nx_graph is expected to have version-agnostic operator names already
    for k, attrs in nx_graph.nodes.items():
        attrs = {k: str(v) for k, v in attrs.items()}
        load_attrs = {k: str(v).strip('"') for k, v in load_graph.nodes[k].items()}
        if 'scope' in load_attrs:
            load_attrs.pop('scope')  # TODO: remove
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
        device = get_model_device(model)

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
    device = get_model_device(model_)
    config = {'input_info': [{"sample_size": sizes} for sizes in input_sample_sizes]}
    input_info_list = create_input_infos(config)
    tensor_list = [create_mock_tensor(info, device) for info in input_info_list]
    args = (tuple(tensor_list),)
    args, _ = sr_wrap_inputs_fn(args, {})
    return nncf_model_output(model_(*args))


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
    ModelDesc("mobilenet_v3_small", torchvision.models.mobilenet_v3_small, [2, 3, 32, 32]),
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
    if torch.cuda.is_available():
        compressed_model.to('cuda')
    compressed_model.do_dummy_forward()
    # internal wrapped model is still in eval mode, switch to the train mode to make sure training graph is ok
    compressed_model.train()
    compressed_model.rebuild_graph()
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

    def get_sparsifiable_modules(self, algo_name):
        # counts wrapped NNCF modules to ignore the ones that are called in the training mode only
        sparsifiable_modules = []
        for module_cls in list(NNCF_MODULES_DICT) + list(NNCF_WRAPPED_USER_MODULES_DICT.values()):
            if algo_name not in module_cls.ignored_algorithms:
                sparsifiable_modules  .append(module_cls.__name__)
        return sparsifiable_modules

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

        config = get_empty_config(input_sample_sizes=desc.input_sample_sizes)
        config["compression"] = {"algorithm": algo}

        compressed_model, compression_ctrl = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn,
                                                      wrap_inputs_fn=desc.wrap_inputs_fn)


        sparsifiable_modules = self.get_sparsifiable_modules(algo)
        ref_num_sparsed = len(get_all_modules_by_type(model, sparsifiable_modules))
        assert ref_num_sparsed == len(compression_ctrl.sparsified_module_info)
        check_model_graph(compressed_model, desc.dot_filename, algo)

    def test_quantize_network(self, desc: ModelDesc, _case_config):
        model = desc.model_builder()

        config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=desc.input_sample_sizes)
        register_bn_adaptation_init_args(config)
        compressed_model, _ = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn,
                                                      wrap_inputs_fn=desc.wrap_inputs_fn)
        check_model_graph(compressed_model, desc.dot_filename, _case_config.graph_dir)

    def test_sparse_quantize_network(self, desc: ModelDesc):
        model = desc.model_builder()

        config = get_empty_config(input_sample_sizes=desc.input_sample_sizes)
        config["compression"] = [
            {"algorithm": "rb_sparsity"},
            {"algorithm": "quantization"}
        ]
        register_bn_adaptation_init_args(config)

        compressed_model, compression_ctrl = \
            create_compressed_model_and_algo_for_test(model, config, dummy_forward_fn=desc.dummy_forward_fn,
                                                      wrap_inputs_fn=desc.wrap_inputs_fn)

        sparsifiable_modules = self.get_sparsifiable_modules('rb_sparsity')
        ref_num_sparsed = len(get_all_modules_by_type(compressed_model, sparsifiable_modules))

        assert ref_num_sparsed == len(compression_ctrl.child_ctrls[0].sparsified_module_info)
        check_model_graph(compressed_model, desc.dot_filename, "quantized_rb_sparsity")


@pytest.mark.skip(reason="Sporadic failures")
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
    config["compression"].update({
        "ignored_scopes": ["GNMT/ResidualRecurrentEncoder[encoder]/Embedding[embedder]",
                           "GNMT/ResidualRecurrentDecoder[decoder]/Embedding[embedder]"]})

    compressed_model = NNCFNetwork(model,
                                   input_infos=create_input_infos(config),
                                   dummy_forward_fn=forward_fn_,
                                   wrap_inputs_fn=gnmt_wrap_inputs_fn,
                                   scopes_without_shape_matching=
                                   ['GNMT/ResidualRecurrentDecoder[decoder]/RecurrentAttention[att_rnn]/'
                                    'BahdanauAttention[attn]'])

    builder = QuantizationBuilder(config, should_init=False)
    builder.apply_to(compressed_model)

    check_model_graph(compressed_model, 'gnmt_variable.dot', _case_config.graph_dir)


def test_resnet18__with_not_qinput(_case_config):
    model = test_models.ResNet18()
    input_shape = [1, 3, 32, 32]

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_shape)
    config["compression"].update({"quantize_inputs": False})
    register_bn_adaptation_init_args(config)

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'resnet18_no_qinput.dot', _case_config.graph_dir)


def test_resnet18__with_ignore(_case_config):
    model = test_models.ResNet18()
    input_shape = [1, 3, 32, 32]

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_shape)
    ignored_scopes = ['{re}ResNet/Sequential\\[layer3\\].*', ]
    config.update({"ignored_scopes": ignored_scopes})  # Global config ignored_scopes for NNCF module replacement
    config["compression"].update({"ignored_scopes": ignored_scopes})  # Local ignored_scopes for quantization
    register_bn_adaptation_init_args(config)

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
                 model_name: str = None, wrap_inputs_fn: Callable = None, model_builder=None, input_info: Dict = None):
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
    SingleLayerModelDesc(model_name='normalize', layer=partial(torch.nn.functional.normalize, p=2),
                         input_sample_sizes=([1, 1, 1, 1], )),

    TensorUnaryMethodsDesc(tensor_method='round'),

    SingleLayerModelDesc(layer=nn.Dropout()),
    SingleLayerModelDesc(layer=nn.Threshold(0.1, 20)),

    SingleLayerModelDesc(layer=nn.BatchNorm1d(1), input_sample_sizes=([2, 1, 1])),
    SingleLayerModelDesc(layer=nn.BatchNorm2d(1), input_sample_sizes=([2, 1, 1, 1])),
    SingleLayerModelDesc(layer=nn.BatchNorm3d(1), input_sample_sizes=([2, 1, 1, 1, 1])),

    SingleLayerModelDesc(layer=nn.GroupNorm(2, 4), input_sample_sizes=([2, 4, 1, 1])),

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
                         input_info=[{"sample_size": [1], "type": "long", "filler": "zeros"},
                                     {"sample_size": [1, 1]},
                                     {"sample_size": [1], "type": "long", "filler": "zeros"}]),

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

    GeneralModelDesc(model_builder=ManyNonEvalModules, input_sample_sizes=([1, 1, 1, 1],)),
    GeneralModelDesc(model_builder=EmbeddingSumModel, input_info={"sample_size": [1, 1],
                                                                  "type": "long",
                                                                  "filler": "zeros"}),
    GeneralModelDesc(model_builder=EmbeddingCatLinearModel, input_info={"sample_size": [1, 1],
                                                                  "type": "long",
                                                                  "filler": "zeros"}),
    GeneralModelDesc(model_builder=MultiOutputSameTensorModel),
    GeneralModelDesc(model_builder=MatMulDivConv, input_sample_sizes=([1, 1, 5, 5], [1, 1, 5, 5])),
    GeneralModelDesc(model_builder=MMDivConv, input_sample_sizes=([5, 5], [5, 5])),
    GeneralModelDesc(model_builder=ConvRelu6HSwishHSigmoid, input_sample_sizes=([1, 1, 5, 5],)),
    GeneralModelDesc(model_builder=ConvBNLeakyReLU, input_sample_sizes=([1, 1, 5, 5],)),
    GeneralModelDesc(model_builder=FC_ConstMul, input_sample_sizes=[1, 3, 6]),
    GeneralModelDesc(model_builder=ConvGeluGetItem, input_sample_sizes=([1, 6, 6],))
]


@pytest.mark.parametrize(
    "synthetic_model_desc", SYNTHETIC_MODEL_DESC_LIST, ids=[m.model_name for m in SYNTHETIC_MODEL_DESC_LIST]
)
def test_synthetic_model_quantization(synthetic_model_desc: IModelDesc):
    config = get_basic_quantization_config(input_sample_sizes=synthetic_model_desc.get_input_sample_sizes(),
                                           input_info=synthetic_model_desc.get_input_info())
    register_bn_adaptation_init_args(config)

    model = synthetic_model_desc.get_model()
    compressed_model, _ = create_compressed_model_and_algo_for_test(
        model, config, wrap_inputs_fn=synthetic_model_desc.get_wrap_inputs_fn())

    check_model_graph(compressed_model, synthetic_model_desc.get_dot_filename(),
                      os.path.join('quantized', 'synthetic_model'))


def test_output_quantization(_case_config):
    model = test_models.UNet()
    input_shape = [1, 3, 360, 480]

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_shape)
    config["compression"].update({"quantize_outputs": True})
    register_bn_adaptation_init_args(config)

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    check_model_graph(compressed_model, 'unet_qoutput.dot', _case_config.graph_dir)


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
    quantization_builder = QuantizationBuilder(config, should_init=False)
    single_config_quantizer_setup = quantization_builder._get_quantizer_setup(compressed_model)
    sketch_graph = compressed_model.get_original_graph()

    potential_quantizer_graph = prepare_potential_quantizer_graph(sketch_graph, single_config_quantizer_setup)
    check_nx_graph(potential_quantizer_graph, desc.dot_filename, _case_dir(hw_config_type.value), sort_dot_graph=False)


def _case_dir(type_hw_config):
    graph_dir = os.path.join('quantized', "hw", type_hw_config)
    return graph_dir


def prepare_potential_quantizer_graph(graph: PTNNCFGraph,
                                      quantizer_setup: SingleConfigQuantizerSetup) -> nx.DiGraph:
    quantizers_weights_attr = {}
    pre_hooked_quantizers_activations_attr = {}  # type: Dict[NNCFNodeName, Tuple[int, str]]
    post_hooked_quantizers_activations_attr = {}  # type: Dict[NNCFNodeName, str]

    # pylint:disable=protected-access
    for qp in quantizer_setup.quantization_points.values():
        if qp.is_weight_quantization_point():
            target_node_name = qp.insertion_point.target_node_name
            qconfig = qp.qconfig
            matching_graph_op_nodes = graph.get_op_nodes_in_scope(graph.get_scope_by_node_name(target_node_name))

            assert len(matching_graph_op_nodes) == 1  # Isn't correct when NNCF module has more than 1 graph node

            module_op_name = matching_graph_op_nodes[0].node_name
            quantizers_weights_attr[module_op_name] = str(qconfig)
        elif qp.is_activation_quantization_point():
            target_node_name = qp.insertion_point.target_node_name
            qconfig = qp.qconfig
            str_qconfig = str(qconfig)
            assert isinstance(qp.insertion_point, ActivationQuantizationInsertionPoint)
            if qp.insertion_point.input_port_id is not None:
                pre_hooked_quantizers_activations_attr[target_node_name] = \
                    (qp.insertion_point.input_port_id, str_qconfig)
            else:
                post_hooked_quantizers_activations_attr[target_node_name] = str_qconfig

    nx_graph = graph.get_graph_for_structure_analysis()
    for nncf_node in graph.get_all_nodes():
        node_key = graph.get_node_key_by_id(nncf_node.node_id)
        node_name = nncf_node.node_name
        if node_name in pre_hooked_quantizers_activations_attr:
            input_port_id, qconf_str = pre_hooked_quantizers_activations_attr[node_name]
            label = "Quantizer: {}".format(qconf_str)
            additional_node_attrs = dict(
                label=label,
                color="purple",
                id=nncf_node.node_id
            )

            pre_hook_quantizer_node_key = node_name + '|IN' + str(input_port_id)
            nx_graph.add_node(pre_hook_quantizer_node_key, **additional_node_attrs)
            # Adding a pre-hook quantizer to a corresponding input port
            edges_with_matching_input_port_id = []

            for edge in graph.get_input_edges(nncf_node):
                from_key = graph.get_node_key_by_id(edge.from_node.node_id)
                to_key = graph.get_node_key_by_id(edge.to_node.node_id)
                if edge.input_port_id == input_port_id:
                    edges_with_matching_input_port_id.append((from_key, to_key))

            assert len(edges_with_matching_input_port_id) == 1
            input_edge_to_break = edges_with_matching_input_port_id[0]

            existing_edge_attrs = deepcopy(nx_graph.edges[input_edge_to_break])
            nx_graph.remove_edge(input_edge_to_break[0], input_edge_to_break[1])
            nx_graph.add_edge(input_edge_to_break[0], pre_hook_quantizer_node_key)
            nx_graph.add_edge(pre_hook_quantizer_node_key, input_edge_to_break[1], **existing_edge_attrs)

        if node_name in post_hooked_quantizers_activations_attr:
            qconf_str = post_hooked_quantizers_activations_attr[node_name]
            label = "Quantizer: {}".format(qconf_str)
            additional_node_attrs = dict(
                label=label,
                color="purple",
                id=nncf_node.node_id
            )

            post_hook_quantizer_node_key = node_name + '|OUT'
            # Adding a post-hook quantizer for the op
            nx_graph.add_node(post_hook_quantizer_node_key, **additional_node_attrs)
            next_nodes = deepcopy(nx_graph._succ[node_key])
            for next_node_name, next_node_attrs in next_nodes.items():
                existing_edge_attrs = deepcopy(next_node_attrs)
                nx_graph.add_edge(post_hook_quantizer_node_key, next_node_name, **existing_edge_attrs)
                nx_graph.remove_edge(node_key, next_node_name)
            nx_graph.add_edge(node_key, post_hook_quantizer_node_key)

        if node_name in quantizers_weights_attr:
            label = "Quantizer: {}".format(quantizers_weights_attr[node_name])
            weight_quantizer_node_key = node_name + '|WEIGHT'
            nx_graph.add_node(weight_quantizer_node_key, label=label, color="purple",
                              id=nncf_node.node_id)
            nx_graph.add_edge(weight_quantizer_node_key, node_key)

    return nx_graph

def test_output_quantization_with_user_forward(_case_config):
    desc = TEST_MODELS_DESC[-1]
    model = desc.model_builder()

    input_shape = desc.input_sample_sizes

    config = get_basic_quantization_config(_case_config.quant_type,
                                            input_sample_sizes=input_shape)
    config["compression"].update({"quantize_outputs": True})
    register_bn_adaptation_init_args(config)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config,
                                                                    dummy_forward_fn=desc.dummy_forward_fn,
                                                                    wrap_inputs_fn=desc.wrap_inputs_fn)
    check_model_graph(compressed_model, 'sr_small_model_qoutput.dot', _case_config.graph_dir)
