#ifndef _DISPATCH_H_
#define _DISPATCH_H_

#include <torch/types.h>

// MSVC cannot even pass __VA_ARGS__ to another macro properly, in contrast with GCC.
// For the DISPATCH_TENSOR_DATA_TYPES macro to work on Windows, had to apply a workaround as described in
// https://renenyffenegger.ch/notes/development/languages/C-C-plus-plus/preprocessor/macros/__VA_ARGS__/index
#define PASS_ON(...) __VA_ARGS__
#define DISPATCH_TENSOR_DATA_TYPES(...) PASS_ON(PASS_ON(AT_DISPATCH_FLOATING_TYPES_AND2)(at::kHalf, at::kBFloat16, __VA_ARGS__))

#endif // _DISPATCH_H_