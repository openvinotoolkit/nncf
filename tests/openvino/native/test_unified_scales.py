from tests.cross_fw.test_templates.test_unified_scales import TemplateTestUnifiedScales
from tests.torch.fx.helpers import get_torch_fx_model_q_transformed
import torch
import openvino as ov

class TestUnifiedScales(TemplateTestUnifiedScales):
    def get_backend_specific_model(self, model: torch.nn.Module) -> ov.Model:
        input_shape = model.INPUT_SHAPE
        backend_model = ov.convert_model(model, example_input=(torch.randn(input_shape), torch.randn(input_shape),))

        return backend_model