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

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from tests.torch.modules.seq2seq.seq2seq_base import PAD


class ResidualRecurrentEncoder(nn.Module):
    """
    Encoder with Embedding, LSTM layers, residual connections and optional
    dropout.

    The first LSTM layer is bidirectional and uses variable sequence length
    API, the remaining (num_layers-1) layers are unidirectional. Residual
    connections are enabled after third LSTM layer, dropout is applied on
    inputs to LSTM layers.
    """

    def __init__(
        self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2, batch_first=False, embedder=None, init_weight=0.1
    ):
        """
        Constructor for the ResidualRecurrentEncoder.

        :param vocab_size: size of vocabulary
        :param hidden_size: hidden size for LSTM layers
        :param num_layers: number of LSTM layers, 1st layer is bidirectional
        :param dropout: probability of dropout (on input to LSTM layers)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param embedder: instance of nn.Embedding, if None constructor will
            create new embedding layer
        :param init_weight: range for the uniform initializer
        """
        super().__init__()
        self.batch_first = batch_first
        self.rnn_layers = nn.ModuleList()
        # 1st LSTM layer, bidirectional
        self.rnn_layers.append(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, batch_first=batch_first, bidirectional=True)
        )

        # 2nd LSTM layer, with 2x larger input_size
        self.rnn_layers.append(
            nn.LSTM((2 * hidden_size), hidden_size, num_layers=1, bias=True, batch_first=batch_first)
        )

        # Remaining LSTM layers
        for _ in range(num_layers - 2):
            self.rnn_layers.append(nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, batch_first=batch_first))

        self.dropout = nn.Dropout(p=dropout)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD)
            nn.init.uniform_(self.embedder.weight.data, -init_weight, init_weight)

    def forward(self, inputs, lengths):
        """
        Execute the encoder.

        :param inputs: tensor with indices from the vocabulary
        :param lengths: vector with sequence lengths (excluding padding)

        returns: tensor with encoded sequences
        """
        x = self.embedder(inputs)

        # bidirectional layer
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths.cpu().numpy(), batch_first=self.batch_first)
        x, _ = self.rnn_layers[0](x)
        x, _ = pad_packed_sequence(x, batch_first=self.batch_first)

        # 1st unidirectional layer
        x = self.dropout(x)
        x, _ = self.rnn_layers[1](x)

        # the rest of unidirectional layers,
        # with residual connections starting from 3rd layer
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[i](x)
            x = x + residual

        return x
