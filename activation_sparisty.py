import numpy as np
import openvino as ov
import nncf

from nncf.experimental.torch.sparsify_activations import sparsify_activations, TargetScope

model = ov.Core().read_model("/home/nsavel/workspace/nncf_local/dummy_models/dummy_llama.xml")

dataset = nncf.Dataset([np.random.random((2, 8)) for _ in range(3)])
sparse_model = sparsify_activations(
    model,
    # dataset=dataset,
    # target_sparsity_by_scope={TargetScope(patterns=[".*linear.*"]): 0.3}
    dataset=nncf.Dataset(np.random.randint(0, 30, (3, 2, 8))),
    target_sparsity_by_scope={
        TargetScope(patterns=[".*gate_proj.*"]): 0.2,
        TargetScope(patterns=[".*up_proj.*"]): 0.3,
        TargetScope(patterns=[".*down_proj.*"]): 0.4,
    }
)
# ov.save_model(sparse_model, "sparse_model.xml", compress_to_fp16=False)
