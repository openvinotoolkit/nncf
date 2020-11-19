
import torch
import torch.nn as nn

import math
import numpy as np
import pandas as pd

from natsort import natsorted
from collections import OrderedDict

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
from torch.autograd import Variable

def to_numpy(var):
    # return var.cpu().data.numpy()
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


def annotate_model_attr(model, input_size, batch_size=-1, repl=False):
    _ = get_model_attr(model, input_size, batch_size, repl)


def get_model_attr(model, input_size, batch_size=-1, repl=False):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            
            module._input_shape=summary[m_key]["input_shape"]
            module._output_shape=summary[m_key]["output_shape"]
            module._nb_params=summary[m_key]["nb_params"]
            
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
            and not module.__class__.__name__ == 'SSDDetectionOutput'
        ):
            hooks.append(module.register_forward_hook(hook))

#     device = device.lower()
#     assert device in [
#         "cuda",
#         "cpu",
#     ], "Input device is not valid, please specify 'cuda' or 'cpu'"

#     if device == "cuda" and torch.cuda.is_available():
    if next(model.parameters()).is_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    dev = next(model.parameters()).device
    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(dev) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if repl:
        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")

        return summary