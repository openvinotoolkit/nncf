"""
 Copyright (c) 2020 Intel Corporation
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
from nncf import NNCFConfig
from tests.test_helpers import  load_exported_onnx_version
from tests.helpers import TwoConvDepthwiseTestModel

from nncf.definitions import NNCF_PACKAGE_ROOT_DIR
import pytest

def get_config_for_export_mode(should_be_onnx_standard=False, should_be_bfp_onnx_fake=False, hw_config_type=None, hw_config_subtype=None) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update({
        "input_info": {
            "sample_size": [1, 1, 4, 4]
        },
        "compression": {
            "algorithm": "quantization",
            "export_to_onnx_standard_ops": should_be_onnx_standard,
            "export_to_onnx_block_floating_point": should_be_bfp_onnx_fake 
        }
    })
    if hw_config_type is not None:
        nncf_config.update({
            "hw_config_type": hw_config_type,
            "hw_config_subtype": hw_config_subtype
        }) 
        
    return nncf_config


@pytest.mark.parametrize('should_be_onnx_standard', (False, True), ids=['fake_quantize','onnx_std'])
@pytest.mark.parametrize('should_be_bfp_onnx_fake', (False, True), ids=['skipBFP','bfp' ])
@pytest.mark.parametrize('bfp', (False, True), ids=['normal','bfp' ])
def test_onnx_export(tmp_path, should_be_onnx_standard, should_be_bfp_onnx_fake, bfp):
    model = TwoConvDepthwiseTestModel()
    hw_config_type = "dla" if bfp else None 
    hw_config_subtype = "int5bfp_dw" if bfp else None 
    nncf_config = get_config_for_export_mode( should_be_onnx_standard=should_be_onnx_standard, 
        should_be_bfp_onnx_fake=should_be_bfp_onnx_fake,
        hw_config_type = hw_config_type,
        hw_config_subtype = hw_config_subtype)
    nncf_config['input_info']={"sample_size": [1,2,4,4]} # 
    onnx_model_proto = load_exported_onnx_version(nncf_config, model,
                                                  path_to_storage_dir=tmp_path)

    node_counters = {'QuantizeLinear':0,
                    'DequantizeLinear':0,
                    'FakeQuantize':0,
                    'FakeQuantizeBfp':0,
                    'Conv':0
                    }

    #pylint:disable=no-member
    for node in onnx_model_proto.graph.node:
        op_type = node.op_type
        print(op_type)

        if op_type in node_counters.keys() :
            node_counters[op_type]  += 1

        expected_results = [[[
                    {
                        'QuantizeLinear':0,
                        'DequantizeLinear':0,
                        'FakeQuantize':4,
                        'FakeQuantizeBfp':0,
                        'Conv':2
                    },
                    {
                        'QuantizeLinear':0,
                        'DequantizeLinear':0,
                        'FakeQuantize':1,
                        'FakeQuantizeBfp':0,
                        'Conv':2
                    }
                ],
                [
                    {
                        'QuantizeLinear':0,
                        'DequantizeLinear':0,
                        'FakeQuantize':4,
                        'FakeQuantizeBfp':0,
                        'Conv':2
                    },
                    {
                        'QuantizeLinear':0,
                        'DequantizeLinear':0,
                        'FakeQuantize':1,
                        'FakeQuantizeBfp':2,
                        'Conv':2
                    }
                ],
            ],
            [
             [
                    {
                        'QuantizeLinear':4,
                        'DequantizeLinear':4,
                        'FakeQuantize':0,
                        'FakeQuantizeBfp':0,
                        'Conv':2
                    },
                    {
                        'QuantizeLinear':1,
                        'DequantizeLinear':1,
                        'FakeQuantize':0,
                        'FakeQuantizeBfp':0,
                        'Conv':2
                    }
                    
                ],
                [
                    {
                        'QuantizeLinear':4,
                        'DequantizeLinear':4,
                        'FakeQuantize':0,
                        'FakeQuantizeBfp':0,
                        'Conv':2
                    },
                    {
                        'QuantizeLinear':1,
                        'DequantizeLinear':1,
                        'FakeQuantize':0,
                        'FakeQuantizeBfp':2,
                        'Conv':2
                    }
                    
                ]
                ]]

    assert node_counters == expected_results[should_be_onnx_standard][should_be_bfp_onnx_fake][bfp]
