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

from tests.torch.modules.seq2seq.decoder import ResidualRecurrentDecoder
from tests.torch.modules.seq2seq.encoder import ResidualRecurrentEncoder
from tests.torch.modules.seq2seq.seq2seq_base import PAD
from tests.torch.modules.seq2seq.seq2seq_base import Seq2Seq


class GNMT(Seq2Seq):
    """
    GNMT v2 model
    """

    def __init__(
        self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2, batch_first=False, share_embedding=True
    ):
        """
        Constructor for the GNMT v2 model.

        :param vocab_size: size of vocabulary (number of tokens)
        :param hidden_size: internal hidden size of the model
        :param num_layers: number of layers, applies to both encoder and
            decoder
        :param dropout: probability of dropout (in encoder and decoder)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param share_embedding: if True embeddings are shared between encoder
            and decoder
        """

        super().__init__(batch_first=batch_first)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD)
            nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size, num_layers, dropout, batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size, num_layers, dropout, batch_first, embedder)

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output
