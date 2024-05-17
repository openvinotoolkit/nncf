import time

import nncf.torch
from nncf.torch.dynamic_graph.patch_pytorch import unpatch_torch_operators, patch_torch_operators
# from optimum.intel import OVModelForFeatureExtraction

import openvino.torch
from transformers import AutoModel, AutoTokenizer
import torch


# import nncf.torch

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
# unpatch_torch_operators()
model = torch.compile(model, backend="openvino")
# patch_torch_operators()
# model = torch.compile(model)

encoded_input = tokenizer(
    ["hello world"], padding=True, truncation=True, return_tensors="pt"
)
with torch.no_grad():
    time.sleep(1)
    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
