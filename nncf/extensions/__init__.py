import torch


class CudaNotAvailableStub:
    def __getattr__(self, item):
        raise RuntimeError("CUDA is not available on this machine. Check that the machine has a GPU and a proper"
                           "driver supporting CUDA {} is installed".format(torch.version.cuda))
