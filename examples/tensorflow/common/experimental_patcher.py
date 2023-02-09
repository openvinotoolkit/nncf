from nncf import NNCFConfig
from nncf.config.utils import is_experimental_quantization


def patch_if_experimental_quantization(nncf_config: NNCFConfig):
    if 'compression' in nncf_config and is_experimental_quantization(nncf_config):
        from nncf.experimental.tensorflow.patch_tf import patch_tf_operations
        patch_tf_operations()
