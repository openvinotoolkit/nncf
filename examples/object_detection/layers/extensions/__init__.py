import os.path

import torch

from torch.utils.cpp_extension import load

ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if torch.cuda.is_available():
    EXTENSIONS = load(
        'extensions', [
            os.path.join(ext_dir, 'extensions.cpp'),
            os.path.join(ext_dir, 'nms/nms.cpp'),
            os.path.join(ext_dir, 'nms/nms_kernel.cpp'),
            os.path.join(ext_dir, 'nms/nms_kernel.cu'),
        ],
        verbose=False
    )
else:
    EXTENSIONS = load(
        'extensions', [
            os.path.join(ext_dir, 'extensions.cpp'),
            os.path.join(ext_dir, 'nms/nms_cpu.cpp'),
            os.path.join(ext_dir, 'nms/nms_kernel.cpp'),
        ],
        verbose=False
    )
