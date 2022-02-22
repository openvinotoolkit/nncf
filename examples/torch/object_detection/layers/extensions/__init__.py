import os.path

import torch

from torch.utils.cpp_extension import load

from nncf.torch.extensions import CudaNotAvailableStub
from nncf.torch.extensions import get_build_directory_for_extension

ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if torch.cuda.is_available():
    EXTENSIONS = load(
        'extensions', [
            os.path.join(ext_dir, 'extensions.cpp'),
            os.path.join(ext_dir, 'nms/nms.cpp'),
            os.path.join(ext_dir, 'nms/nms_kernel.cu'),
        ],
        verbose=False,
        build_directory=get_build_directory_for_extension('extensions')
    )
else:
    EXTENSIONS = CudaNotAvailableStub
