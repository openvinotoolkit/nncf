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
import copy
import logging
import os
import sys
from collections import namedtuple
from functools import partial
from typing import List, Tuple

import onnx
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

from nncf.torch import nncf_model_input
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.layers import ITERATION_MODULES
from nncf.torch.layers import NNCF_RNN
from nncf.torch.layers import LSTMCellNNCF
from nncf.torch.model_creation import create_compressed_model
from nncf.torch.nncf_module_replacement import collect_modules_and_scopes_by_predicate
from nncf.torch.utils import get_model_device
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import get_grads
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.modules.seq2seq.gnmt import GNMT


def replace_lstm(model):
    def replace_fn(module_):
        if not isinstance(module_, nn.LSTM):
            return module_
        device = get_model_device(module_)
        custom_lstm = NNCF_RNN(
            "LSTM",
            input_size=module_.input_size,
            hidden_size=module_.hidden_size,
            num_layers=module_.num_layers,
            bidirectional=module_.bidirectional,
            batch_first=module_.batch_first,
            dropout=module_.dropout,
            bias=module_.bias,
        )

        def get_param_names(bias: bool) -> List[str]:
            suffixes = ["ih", "hh"]
            names = ["weight_" + suffix for suffix in suffixes]
            if bias:
                names += ["bias_" + suffix for suffix in suffixes]
            return names

        for layer_idx in range(custom_lstm.num_layers):
            for d in range(custom_lstm.num_directions):
                for name in get_param_names(custom_lstm.bias):
                    suffix = "_reverse" if d == 1 else ""
                    param_name = name + "_l{}{}".format(layer_idx, suffix)
                    param = getattr(module_, param_name)
                    getattr(custom_lstm, param_name).data.copy_(param.data)
        custom_lstm.to(device)
        return custom_lstm

    lstm_modules = collect_modules_and_scopes_by_predicate(model, lambda x: isinstance(x, nn.LSTM))
    if isinstance(model, nn.LSTM):
        return replace_fn(model)
    from nncf.torch.nncf_module_replacement import _replace_module_by_scope

    for module, scope_set in lstm_modules.items():
        replaced_module = replace_fn(module)
        for scope in scope_set:
            _replace_module_by_scope(model, scope, replaced_module)
    return model


def clone_test_data(data_list) -> List[torch.Tensor]:
    results = []
    x = data_list[0]
    result = x if isinstance(x, PackedSequence) else x.clone()
    results.append(result)
    for tensor_list in data_list[1:]:
        result = ()
        for tensor in tensor_list:
            if isinstance(tensor, Variable):
                sub_result = tensor.data.clone()
                sub_result = Variable(sub_result, requires_grad=True)
            else:
                sub_result = tensor.clone()
            result += (sub_result,)
        results.append(result)
    return results


LSTMTestSizes = namedtuple("LSTMTestSizes", ["input_size", "hidden_size", "batch", "seq_length"])
LSTMTestData = namedtuple("LSTMTestData", ["x", "h0", "c0", "weight_ih", "weight_hh", "bias_ih", "bias_hh"])


@pytest.mark.parametrize(
    "sizes",
    [LSTMTestSizes(512, 768, 128, 50), LSTMTestSizes(3, 3, 3, 3), LSTMTestSizes(1, 1, 1, 1)],
    ids=lambda val: "[{}]".format("-".join([str(v) for v in val])),
)
class TestLSTMCell:
    @staticmethod
    def generate_lstm_data(
        p: LSTMTestSizes,
        num_layers: int = 1,
        num_directions: int = 1,
        variable_length: bool = False,
        sorted_: bool = True,
        batch_first: bool = True,
        use_cuda: bool = False,
        bias: bool = True,
        empty_initial: bool = False,
        is_backward: bool = False,
    ) -> LSTMTestData:
        num_chunks = 4
        seq_list = []
        if variable_length:
            seq_lens = torch.IntTensor(p.batch).random_(1, p.seq_length + 1)
            if sorted_:
                seq_lens = torch.sort(seq_lens, descending=True).values
            for seq_size in seq_lens:
                seq_list.append(torch.randn(seq_size.item(), p.input_size))
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=batch_first)
            x_data = torch.nn.utils.rnn.pack_padded_sequence(
                padded_seq_batch, lengths=seq_lens, batch_first=batch_first, enforce_sorted=sorted_
            )

        else:
            size = (p.seq_length, p.batch, p.input_size)
            if batch_first:
                size = (p.batch, p.seq_length, p.input_size)
            x_data = torch.randn(*size)

        def wrap_tensor(tensor):
            wrapped = tensor
            if use_cuda:
                wrapped = wrapped.cuda()
            if is_backward:
                wrapped = Variable(wrapped, requires_grad=True)
            return wrapped

        if use_cuda:
            x_data = x_data.cuda()
        h0, c0, wih, whh, bih, bhh = ([] for _ in range(6))
        for layer_ in range(num_layers):
            for _ in range(num_directions):
                layer_input_size = p.input_size if layer_ == 0 else p.hidden_size * num_directions
                if not empty_initial:
                    h0.append(wrap_tensor(torch.randn(p.batch, p.hidden_size)))
                    c0.append(wrap_tensor(torch.randn(p.batch, p.hidden_size)))
                wih.append(wrap_tensor(torch.rand(num_chunks * p.hidden_size, layer_input_size)))
                whh.append(wrap_tensor(torch.rand(num_chunks * p.hidden_size, p.hidden_size)))
                if bias:
                    bih.append(wrap_tensor(torch.rand(num_chunks * p.hidden_size)))
                    bhh.append(wrap_tensor(torch.rand(num_chunks * p.hidden_size)))
        result = LSTMTestData(x_data, h0, c0, wih, whh, bih, bhh)
        return result

    @staticmethod
    def set_weights(cell: nn.LSTMCell, data: LSTMTestData):
        for name in TestLSTM.get_param_names(bias=True):
            param = getattr(data, name)
            if param:
                getattr(cell, name).data.copy_(param[0].data)

    def test_forward_lstm_cell(self, sizes, _seed):
        p = sizes
        ref_data = TestLSTMCell.generate_lstm_data(p, batch_first=False)
        test_data = LSTMTestData(*clone_test_data(ref_data))

        ref_rnn = nn.LSTMCell(p.input_size, p.hidden_size)
        TestLSTMCell.set_weights(ref_rnn, ref_data)
        test_rnn = LSTMCellNNCF(p.input_size, p.hidden_size)
        TestLSTMCell.set_weights(test_rnn, test_data)

        for i in range(p.seq_length):
            ref_result = ref_rnn(ref_data.x[i], (ref_data.h0[0], ref_data.c0[0]))
            test_result = test_rnn(test_data.x[i], (test_data.h0[0], test_data.c0[0]))
            for ref, test in list(zip(ref_result, test_result)):
                torch.testing.assert_close(test, ref)

    def test_backward_lstm_cell(self, sizes, _seed):
        p = sizes
        ref_data = TestLSTMCell.generate_lstm_data(p, batch_first=False, is_backward=True)
        with torch.no_grad():
            test_data = LSTMTestData(*clone_test_data(ref_data))

        ref_rnn = nn.LSTMCell(p.input_size, p.hidden_size)
        TestLSTMCell.set_weights(ref_rnn, ref_data)
        test_rnn = LSTMCellNNCF(p.input_size, p.hidden_size)
        TestLSTMCell.set_weights(test_rnn, test_data)

        for i in range(p.seq_length):
            ref_result = ref_rnn(ref_data.x[i], (ref_data.h0[0], ref_data.c0[0]))
            test_result = test_rnn(test_data.x[i], (test_data.h0[0], test_data.c0[0]))
            ref_result[0].sum().backward()
            test_result[0].sum().backward()
            ref_grads = get_grads([ref_data.h0[0], ref_data.c0[0]])
            ref_grads += get_grads([ref_rnn.weight_ih, ref_rnn.weight_hh, ref_rnn.bias_ih, ref_rnn.bias_hh])
            test_grads = get_grads([ref_data.h0[0], ref_data.c0[0]])
            test_grads += get_grads([test_rnn.weight_ih, test_rnn.weight_hh, test_rnn.bias_ih, test_rnn.bias_hh])
            for ref, test in list(zip(test_grads, ref_grads)):
                torch.testing.assert_close(test, ref)


def test_export_lstm_cell(tmp_path):
    config = get_empty_config(model_size=1, input_sample_sizes=[1, 1])
    config["compression"] = {"algorithm": "quantization"}
    register_bn_adaptation_init_args(config)

    model, algo = create_compressed_model_and_algo_for_test(LSTMCellNNCF(1, 1), config)

    test_path = str(tmp_path.joinpath("test.onnx"))
    # Exporting the operator ::chunk to ONNX opset version 9 is not supported.
    # Support for this operator was added in version 11
    algo.export_model(test_path, save_format="onnx_11")
    assert os.path.exists(test_path)

    onnx_num = 0
    model = onnx.load(test_path)

    for node in model.graph.node:
        if node.op_type == "FakeQuantize":
            onnx_num += 1
    assert onnx_num == 11


@pytest.mark.parametrize(
    "sizes",
    [LSTMTestSizes(512, 324, 128, 50), LSTMTestSizes(3, 3, 3, 3), LSTMTestSizes(1, 1, 1, 1)],
    ids=lambda val: "[{}]".format("-".join([str(v) for v in val])),
)
@pytest.mark.parametrize("bidirectional", (True, False), ids=("bi", "uni"))
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("num_layers", [1, 2], ids=["single_layer", "stacked"])
@pytest.mark.parametrize("batch_first", [True, False], ids=["batch_first", "seq_first"])
@pytest.mark.parametrize(
    ("variable_length", "sorted_"),
    ([True, True], [True, False], [False, False]),
    ids=["packed_sorted", "packed_unsorted", "not_packed"],
)
@pytest.mark.parametrize("empty_initial", [True, False], ids=["no_initial", "with_initial"])
# TODO: dropout gives different result. Looks like different random seed on CPU
# @pytest.mark.parametrize('dropout', [0, 0.9], ids=['no_dropout', 'with_dropout'])
@pytest.mark.parametrize("dropout", [0], ids=["no_dropout"])
class TestLSTM:
    def test_forward_lstm(
        self,
        sizes,
        bidirectional,
        num_layers,
        bias,
        batch_first,
        variable_length,
        sorted_,
        use_cuda,
        empty_initial,
        dropout,
        _seed,
    ):
        if not torch.cuda.is_available() and use_cuda is True:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        num_directions = 2 if bidirectional else 1
        p = sizes

        ref_data = TestLSTMCell.generate_lstm_data(
            p, num_layers, num_directions, variable_length, sorted_, batch_first, use_cuda, bias, empty_initial
        )

        ref_rnn = nn.LSTM(
            input_size=p.input_size,
            hidden_size=p.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
            bias=bias,
            dropout=dropout,
        )
        self.set_ref_lstm_weights(ref_data, ref_rnn, num_layers, num_directions, bias)
        ref_hidden = None if empty_initial else self.get_ref_lstm_hidden(ref_data)

        test_data = LSTMTestData(*clone_test_data(ref_data))

        class ModelWrapper(nn.Module):
            def __init__(self, lstm):
                super().__init__()
                self.lstm = lstm

            def forward(self, *input_):
                return self.lstm(*input_)

        wrapped_ref_rnn = ModelWrapper(ref_rnn)
        wrapped_test_rnn = replace_lstm(copy.deepcopy(wrapped_ref_rnn))
        test_rnn = wrapped_test_rnn.lstm
        test_hidden = None if empty_initial else self.get_test_lstm_hidden(test_data)

        if use_cuda:
            ref_rnn.cuda()
            test_rnn.cuda()
        ref_output, (ref_hn, ref_cn) = ref_rnn(ref_data.x, ref_hidden)
        test_output, (test_hn, test_cn) = test_rnn(test_data.x, test_hidden)

        torch.testing.assert_close(test_hn[0], ref_hn[0], rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(test_cn[0], ref_cn[0], rtol=1e-3, atol=1e-4)
        if variable_length:
            torch.testing.assert_close(test_output.batch_sizes, ref_output.batch_sizes)
            torch.testing.assert_close(test_output.data, ref_output.data, rtol=1e-2, atol=1e-3)
            if not sorted_:
                torch.testing.assert_close(test_output.sorted_indices, ref_output.sorted_indices)
                torch.testing.assert_close(test_output.unsorted_indices, ref_output.unsorted_indices)
        else:
            torch.testing.assert_close(test_output, ref_output, rtol=9e-2, atol=15e-4)

    def test_backward_lstm(
        self,
        sizes,
        bidirectional,
        num_layers,
        bias,
        batch_first,
        variable_length,
        sorted_,
        use_cuda,
        empty_initial,
        dropout,
        _seed,
    ):
        if not torch.cuda.is_available() and use_cuda is True:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        num_directions = 2 if bidirectional else 1

        p = sizes

        ref_data = TestLSTMCell.generate_lstm_data(
            p, num_layers, num_directions, variable_length, sorted_, batch_first, use_cuda, bias, empty_initial, True
        )

        ref_rnn = nn.LSTM(
            input_size=p.input_size,
            hidden_size=p.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
            bias=bias,
            dropout=dropout,
        )
        self.set_ref_lstm_weights(ref_data, ref_rnn, num_layers, num_directions, bias)
        ref_hidden = None if empty_initial else self.get_ref_lstm_hidden(ref_data)

        test_data = LSTMTestData(*clone_test_data(ref_data))
        test_rnn = replace_lstm(copy.deepcopy(ref_rnn))
        test_hidden = None if empty_initial else self.get_test_lstm_hidden(test_data)

        if use_cuda:
            ref_rnn.cuda()
            test_rnn.cuda()

        ref_output, _ = ref_rnn(ref_data.x, ref_hidden)
        test_output, _ = test_rnn(test_data.x, test_hidden)

        ref_output[0].sum().backward()
        test_output[0].sum().backward()

        ref_grads = get_grads(self.flatten_nested_lists(ref_rnn.all_weights))
        test_grads = get_grads(self.flatten_nested_lists(test_rnn.all_weights))
        if not empty_initial:
            # TODO: compare gradient of all hidden
            ref_grads += get_grads([ref_data.h0[0], ref_data.c0[0]])
            test_grads += get_grads([test_hidden[0][0], test_hidden[1][0]])
        for ref, test in list(zip(test_grads, ref_grads)):
            torch.testing.assert_close(test, ref, rtol=1e-1, atol=1e-1)

    @classmethod
    def flatten_nested_lists(cls, nested_list: List) -> List[torch.Tensor]:
        return [tensor for tensor_tuple in nested_list for tensor in tensor_tuple]

    @classmethod
    def get_test_lstm_hidden(cls, data: LSTMTestData) -> List[Tuple[torch.Tensor, ...]]:
        result = []
        hidden_names = ["h0", "c0"]
        for name in hidden_names:
            hidden_list = getattr(data, name)
            element = ()
            num_hidden = len(hidden_list)
            for i in range(num_hidden):
                element += (hidden_list[i],)
            result.append(element)
        return result

    @classmethod
    def get_ref_lstm_hidden(cls, data: LSTMTestData) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = cls.get_test_lstm_hidden(data)
        hidden_states = [torch.unsqueeze(tensor, dim=0) for tensor in hidden[0]]
        cell_states = [torch.unsqueeze(tensor, dim=0) for tensor in hidden[1]]
        return (torch.cat(hidden_states, dim=0), torch.cat(cell_states, dim=0))

    @classmethod
    def set_ref_lstm_weights(
        cls, data: LSTMTestData, nn_lstm: nn.LSTM, num_layers: int, num_directions: int, bias: bool
    ):
        for layer_idx in range(num_layers):
            for d in range(num_directions):
                i = layer_idx * num_directions + d
                for name in cls.get_param_names(bias):
                    suffix = "_reverse" if d == 1 else ""
                    param = getattr(data, name)
                    param_name = name + "_l{}{}".format(layer_idx, suffix)
                    getattr(nn_lstm, param_name).data.copy_(param[i].data)

    @classmethod
    def get_param_names(cls, bias: bool) -> List[str]:
        suffixes = ["ih", "hh"]
        names = ["weight_" + suffix for suffix in suffixes]
        if bias:
            names += ["bias_" + suffix for suffix in suffixes]
        return names


def test_export_stacked_bi_lstm(tmp_path):
    p = LSTMTestSizes(3, 3, 3, 3)
    config = get_empty_config(input_sample_sizes=[1, p.hidden_size, p.input_size])
    config["compression"] = {"algorithm": "quantization"}
    register_bn_adaptation_init_args(config)

    # TODO: batch_first=True fails with building graph: ambiguous call to mul or sigmoid
    test_rnn = NNCF_RNN(
        "LSTM", input_size=p.input_size, hidden_size=p.hidden_size, num_layers=2, bidirectional=True, batch_first=False
    )
    model, algo = create_compressed_model_and_algo_for_test(test_rnn, config)

    test_path = str(tmp_path.joinpath("test.onnx"))
    # Exporting the operator ::chunk to ONNX opset version 9 is not supported.
    # Support for this operator was added in version 11
    algo.export_model(test_path, save_format="onnx_11")
    assert os.path.exists(test_path)

    onnx_num = 0

    model = onnx.load(test_path)
    for node in model.graph.node:
        if node.op_type == "FakeQuantize":
            onnx_num += 1
    assert onnx_num == 42


class TestNumberOfNodes:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    def test_number_of_calling_fq_for_lstm(self):
        p = LSTMTestSizes(1, 1, 1, 5)
        num_layers = 2
        bidirectional = True
        num_directions = 2 if bidirectional else 1
        bias = True
        batch_first = False
        config = get_empty_config(input_sample_sizes=[p.seq_length, p.batch, p.input_size])
        config["compression"] = {"algorithm": "quantization", "quantize_inputs": True}
        register_bn_adaptation_init_args(config)

        test_data = TestLSTMCell.generate_lstm_data(p, num_layers, num_directions, bias=bias, batch_first=batch_first)

        test_rnn = NNCF_RNN(
            "LSTM",
            input_size=p.input_size,
            hidden_size=p.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            bias=bias,
            batch_first=batch_first,
        )
        TestLSTM.set_ref_lstm_weights(test_data, test_rnn, num_layers, num_directions, bias)
        test_hidden = TestLSTM.get_test_lstm_hidden(test_data)

        model, algo = create_compressed_model_and_algo_for_test(test_rnn, config)

        class Counter:
            def __init__(self):
                self.count = 0

            def next(self):
                self.count += 1

        def hook(model, input_, counter):
            counter.next()

        counters = {}
        counter_for_input_quantizer = None
        inter_layer_reset_point_post_aq_counters = {}
        for name, quantizer in algo.all_quantizations.items():
            counter = Counter()
            quantizer.register_forward_pre_hook(partial(hook, counter=counter))
            if str(name) == "/nncf_model_input_0|OUTPUT":
                counter_for_input_quantizer = counter
                continue
            if "RNNResetPoint" in str(name):
                inter_layer_reset_point_post_aq_counters[name] = counter
                continue
            counters[name] = counter
        _ = model(test_data.x, test_hidden)

        # NB: below may always fail in debug due to superfluous 'cat' nodes
        assert model.nncf.get_graph().get_nodes_count() == 120
        assert len(counters) + 2 == 42  # 8 WQ + 32 AQ + 1 input AQ + 1 reset point AQ
        for counter in counters.values():
            assert counter.count == p.seq_length
        assert counter_for_input_quantizer.count == 1
        for counter in inter_layer_reset_point_post_aq_counters.values():
            assert counter.count == 1

    @pytest.mark.skip(reason="Sporadic failures")
    def test_number_of_calling_fq_for_gnmt(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        batch_first = False
        vocab_size = 32000
        model_config = {
            "hidden_size": 100,
            "vocab_size": vocab_size,
            "num_layers": 4,
            "dropout": 0.2,
            "batch_first": batch_first,
            "share_embedding": True,
        }
        batch_size = 128
        sequence_size = 50
        input_sample_size = [batch_size, sequence_size] if batch_first else [sequence_size, batch_size]
        config = get_empty_config(input_sample_sizes=input_sample_size)
        config["compression"] = {"algorithm": "quantization", "quantize_inputs": True}
        config["scopes_without_shape_matching"] = [
            "GNMT/ResidualRecurrentDecoder[decoder]/RecurrentAttention[att_rnn]/BahdanauAttention[attn]",
        ]
        register_bn_adaptation_init_args(config)

        model = GNMT(**model_config)
        model = replace_lstm(model)
        model.to(device)

        def dummy_forward_fn(model, seq_len=sequence_size):
            def gen_packed_sequence():
                seq_list = []
                seq_lens = torch.LongTensor(batch_size).random_(1, seq_len + 1)
                seq_lens = torch.sort(seq_lens, descending=True).values
                for seq_size in seq_lens:
                    seq_list.append(torch.LongTensor(seq_size.item()).random_(1, vocab_size).to(device))
                padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=batch_first)
                return padded_seq_batch, seq_lens

            x_data, seq_lens = gen_packed_sequence()
            input_encoder = x_data
            input_enc_len = seq_lens.to(device)
            input_decoder = gen_packed_sequence()[0]
            wrap_nncf_model_outputs_with_objwalk(model(input_encoder, input_enc_len, input_decoder))

        def gnmt_wrap_inputs_fn(model_args, model_kwargs):
            # Assuming 3 args to wrap: input_encoder, input_enc_len, input_decoder, and 0 kwargs to wrap
            model_args = (
                nncf_model_input(model_args[0]),
                nncf_model_input(model_args[1]),
                nncf_model_input(model_args[2]),
            )
            return model_args, model_kwargs

        algo, model = create_compressed_model(
            model, config, dummy_forward_fn=dummy_forward_fn, wrap_inputs_fn=gnmt_wrap_inputs_fn, dump_graphs=False
        )
        model.to(device)

        class Counter:
            def __init__(self):
                self.count = 0

            def next(self):
                self.count += 1

        def hook(model, input_, counter):
            counter.next()

        counters = {}
        for name, quantizer in algo.all_quantizations.items():
            counter = Counter()
            counters[str(name)] = counter
            quantizer.register_forward_pre_hook(partial(hook, counter=counter))
        dummy_forward_fn(model)

        assert (
            model.nncf.get_graph().get_nodes_count() == 370
        )  # NB: may always fail in debug due to superfluous 'cat' nodes
        assert len(counters) == 136
        ref_call_counts = {
            "cell": sequence_size,
            "LSTMCellForwardNNCF": sequence_size,
            # embedding module is shared between the decoder and encoder,
            # associated weight quantizer will be called twice
            "embedding": 2,
            # unified scales for 4 FQ
            "NNCF_RNN[0]/StackedRNN[rnn_impl]/StackedRNNResetPoint/cat_0|OUTPUT": 4,
        }
        for name, counter in counters.items():
            print(name, counter.count)
            for ref_key, ref_count in ref_call_counts.items():
                if ref_key in name:
                    assert counter.count == ref_count, name
                    break
        new_seq_len = int(sequence_size / 2)
        dummy_forward_fn(model, new_seq_len)

        ref_call_counts = {
            "cell": sequence_size + new_seq_len,
            "LSTMCellForwardNNCF": sequence_size + new_seq_len,
            "embedding": 4,
            "NNCF_RNN[0]/StackedRNN[rnn_impl]/StackedRNNResetPoint/cat_0|OUTPUT": 8,
        }
        assert model.nncf.get_graph().get_nodes_count() == 370
        assert len(counters) == 136
        for name, counter in counters.items():
            for ref_key, ref_count in ref_call_counts.items():
                if ref_key in name:
                    assert counter.count == ref_count, name
                    break

    def test_number_of_nodes_for_module_in_loop(self):
        num_iter = 5

        class LoopModule(nn.Module):
            @ITERATION_MODULES.register("Inner")
            class Inner(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.operator1 = torch.sigmoid
                    self.operator2 = torch.tanh

                def forward(self, x):
                    s = self.operator1(x)
                    t = self.operator2(x)
                    result = t + s
                    return result

                @staticmethod
                def nodes_number():
                    return 3

            def __init__(self):
                super().__init__()
                self.inner = self.Inner()

            def forward(self, x):
                for _ in range(num_iter):
                    x = self.inner(x)
                return x

            def nodes_number(self):
                return self.inner.nodes_number()

        test_module = LoopModule()
        context = TracingContext()
        context.enable_trace_dynamic_graph()
        with context as ctx:
            _ = test_module(torch.zeros(1))
            assert ctx.graph.get_nodes_count() == test_module.nodes_number()

    def test_number_of_nodes_for_module_in_loop__not_input_node(self):
        num_iter = 5

        class LoopModule(nn.Module):
            class Inner(nn.Module):
                def forward(self, x):
                    s = F.sigmoid(x)
                    t = F.tanh(x)
                    result = F.sigmoid(x) * t + F.tanh(x) * s
                    return result

                @staticmethod
                def nodes_number():
                    return 7

            def __init__(self):
                super().__init__()
                self.inner = self.Inner()

            def forward(self, x):
                for _ in range(num_iter):
                    x = self.inner(F.relu(x))
                return x

            def nodes_number(self):
                return self.inner.nodes_number() + num_iter

        test_module = LoopModule()
        context = TracingContext()
        context.enable_trace_dynamic_graph()
        with context as ctx:
            _ = test_module(torch.zeros(1))
            assert ctx.graph.get_nodes_count() == test_module.nodes_number()

    def test_number_of_nodes_for_module_with_nested_loops(self):
        num_iter = 5

        class TestIterModule(nn.Module):
            @ITERATION_MODULES.register()
            class TestIterModule_ResetPoint(nn.Module):
                def __init__(self, loop_module):
                    super().__init__()
                    self.loop_module = loop_module

                def forward(self, x):
                    return self.loop_module(F.relu(x))

            def __init__(self):
                super().__init__()
                self.loop_module = self.LoopModule2()
                self.reset_point = self.TestIterModule_ResetPoint(self.loop_module)

            def forward(self, x):
                for _ in range(num_iter):
                    x = self.reset_point(x)
                return x

            class LoopModule2(nn.Module):
                @ITERATION_MODULES.register()
                class LoopModule2_ResetPoint(nn.Module):
                    def __init__(self, inner):
                        super().__init__()
                        self.inner = inner

                    def forward(self, x):
                        return self.inner(F.relu(x))

                def __init__(self):
                    super().__init__()
                    self.inner = self.Inner()
                    self.reset_helper = self.LoopModule2_ResetPoint(self.inner)

                def forward(self, x):
                    for _ in range(num_iter):
                        self.reset_helper(x)
                    return x

                class Inner(nn.Module):
                    def forward(self, x):
                        s = F.sigmoid(x)
                        t = F.tanh(x)
                        result = t + s
                        return result

        test_module = TestIterModule()
        context = TracingContext()
        context.enable_trace_dynamic_graph()
        with context as ctx:
            _ = test_module(torch.zeros(1))
            assert ctx.graph.get_nodes_count() == num_iter

    def test_number_of_nodes_for_repeated_module(self):
        class LoopModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.operator = F.relu
                self.layers = nn.ModuleList([nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)])

            def forward(self, x):
                for layer in self.layers:
                    x = F.relu(layer(x))
                return x

        test_module = LoopModule()
        context = TracingContext()
        context.enable_trace_dynamic_graph()
        with context as ctx:
            x = test_module(torch.zeros(1, 1, 1, 1))
            assert ctx.graph.get_nodes_count() == 4  # NB: may always fail in debug due to superfluous 'cat' nodes
            _ = test_module(x)
            assert ctx.graph.get_nodes_count() == 8  # NB: may always fail in debug due to superfluous 'cat' nodes
