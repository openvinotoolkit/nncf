### This file must be placed in the directory of transformers repo

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

import nncf
from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch.search_blocks import get_building_blocks
from nncf.torch.search_blocks import get_all_modules_per_blocks
from nncf.torch.search_blocks import get_all_node_op_addresses_in_block

d = {"architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.9.1",
  "type_vocab_size": 2,
  "use_cache": True,
  "vocab_size": 21128
}
config = BertConfig(**d)

model = BertForSequenceClassification(config)
nncf_config = NNCFConfig()
nncf_config.update({
        "model": "BertForSequenceClassification",
        "input_info": [
        {
            "sample_size": [1, 128],
            "type": "long"
        },
        {
            "sample_size": [1, 128],
            "type": "long"
        },
        {
            "sample_size": [1, 128],
            "type": "long"
        }
        ],
    })

compression_algo_controller, compressed_model = create_compressed_model(model, nncf_config)
#print(model)
#print(compressed_model)

building_blocks  = get_building_blocks(compressed_model, allow_nested_blocks=False)
node_op_address_per_block = get_all_node_op_addresses_in_block(compressed_model, building_blocks)
modules_per_blocks = get_all_modules_per_blocks(compressed_model, building_blocks)
#print(building_blocks)
#print(node_op_address_per_block)
#print(modules_per_blocks)
