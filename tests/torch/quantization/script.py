

import numpy as np
def old_fq(x, inp_low, inp_hi, levels=255):
    assert inp_hi > inp_low
    s = (levels - 1) / (inp_hi - inp_low)
    q = min(max(x, inp_low), inp_hi) - inp_low
    q *= s
    q = np.round(q)
    dq = q / s + inp_low
    return q, dq

def new_fq(x, inp_low, inp_hi, levels=255):
    assert inp_hi > inp_low
    s = (levels - 1) / (inp_hi - inp_low)
    print(f'quant_zero {-inp_low * s}')
    q_zero = np.round(-inp_low * s)
    q = min(max(x, inp_low), inp_hi) - inp_low
    q *= s
    q -= q_zero
    q = np.round(q)
    dq = q / s
    return q, dq

def calc_diff(inp_low, inp_hi, levels=255):
    s = (levels - 1) / (inp_hi - inp_low)
    diff = inp_low * s
    diff = diff - np.round(diff)
    diff /= s
    return diff

from tests.torch.quantization.test_functions import ReferenceQuantize
inp_low, inp_hi = np.array([-1.]), np.array([3.])
levels = 255
new_input_low, new_input_range = ReferenceQuantize.tune_range(inp_low, inp_hi - inp_low, levels)
print(f'diff: {calc_diff(new_input_low, new_input_range + new_input_low)}')
for x in range(5):
    old_res = old_fq(x / 2, new_input_low, new_input_range + new_input_low, levels=levels)
    new_res = new_fq(x / 2, new_input_low, new_input_range + new_input_low, levels=levels)
    print(f'Value {x / 2}, old_res {old_res}, new_res {new_res}, actual diff {old_res[1] - new_res[1]}')
