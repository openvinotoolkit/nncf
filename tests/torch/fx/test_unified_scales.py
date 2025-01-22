from tests.cross_fw.test_templates.test_unified_scales import TemplateTestUnifiedScales
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.fx.helpers import get_torch_fx_model_q_transformed
import torch

class TestUnifiedScales(TemplateTestUnifiedScales):
    def get_backend_specific_model(self, model: torch.nn.Module) -> NNCFNetwork:
        input_shape = model.INPUT_SHAPE
        backend_model = get_torch_fx_model_q_transformed(model, (torch.randn(input_shape), torch.randn(input_shape),))

        return backend_model