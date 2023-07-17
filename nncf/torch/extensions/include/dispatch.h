#ifndef _DISPATCH_H_
#define _DISPATCH_H_

#include <torch/types.h>

#define DISPATCH_TENSOR_DATA_TYPES(...) AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, __VA_ARGS__)

#endif // _DISPATCH_H_