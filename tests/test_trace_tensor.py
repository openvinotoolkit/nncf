import torch
from nncf.dynamic_graph.trace_tensor import TracedTensor, TensorMeta, pass_meta_from_tensor


def test_trace_non_in_place_ops():
    tensor_ = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    meta = TensorMeta(None, 0, tensor_.shape)
    traced_tensor = TracedTensor.from_torch_tensor(tensor_, meta)

    traced_tensor = traced_tensor.view((1, 4))
    assert isinstance(traced_tensor, TracedTensor)

    non_traced_tensor = traced_tensor.exp()
    assert torch.is_tensor(non_traced_tensor)

    traced_tensor = pass_meta_from_tensor(non_traced_tensor, traced_tensor)
    assert isinstance(traced_tensor, TracedTensor)
