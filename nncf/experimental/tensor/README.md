# NNCF wrapper of the Tensors

The `nncf.Tensor` class is a wrapper class that provides a common interface for different types of tensors,
such as NumPy and PyTorch. This allows algorithms to be written that are abstracted from the underlying model type,
making them more portable and reusable.

## Usage

The main idea is common algorithms should use wrapped tensors and provide to backend-specific function unwrapped tensor.

### Initialization Tensor

```python
import nncf

import numpy as np
numpy_array = np.array([1,2])
nncf_tensor = nncf.Tensor(numpy_array)

import torch
torch_tensor = np.array([1,2])
nncf_tensor = nncf.Tensor(torch_tensor)
```

### Math operations

All math operations are overrided to operated with wrapped object and return `nncf.Tensor`

```python
tensor_a = nncf.Tensor(np.array([1,2]))
tenor_b = nncf.Tensor(np.array([1,2]))
tensor_a + tenor_b  # nncf.Tensor(array([2, 4]))
```

### Math operations

All math operations are overrided to operated with wrapped object and return `nncf.Tensor`

```python
tensor_a = nncf.Tensor(np.array([1,2]))
tenor_b = nncf.Tensor(np.array([1,2]))
tensor_a + tenor_b  # nncf.Tensor(array([2, 4]))
```

### Comparison operators

All math operations are overrided to operated with wrapped object and return `nncf.Tensor`

```python
tensor_a = nncf.Tensor(np.array([1,2]))
tenor_b = nncf.Tensor(np.array([1,2]))
tensor_a < tenor_b  # nncf.Tensor(array([False, False]))
```

### Method of the Tensor class

Some methods of the tensors available from wrapped nncf.Tensor class like `max`, `flatten` and other common methods.

```python
nncf_tensor.max()  # nncf.Tensor(2)
```

### Functions over Tensor

```python
nncf.max(nncf_tensor)  # nncf.Tensor(2)
```

### Loop over Tensor

For `nncf.Tensor` available `TensorIterator` that return `nncf.Tensor`

```python
tensor_a = nncf.Tensor(np.array([1,2]))
for x in tensor_a:
    print(x)

# nncf.Tensor(1)
# nncf.Tensor(2)
```

### Get element by index Tensor

```python
tensor_a = nncf.Tensor(np.array([[1],[2]]))
tensor_a[0]    # nncf.Tensor(array([1]))
tensor_a[0:2]  # nncf.Tensor(array([[1],[2]]))
```

## Class feature enhancement

**NOTE** Use names and descriptions in numpy style.

### Add new method or function

1. Add method to [class Tensor](tensor.py)

    ```python
    class Tensor:
        ...
        def foo(self, arg1: Type) -> "Tensor":
            return functions.foo(self, arg1)
    ```

2. Add function to [function.py](function.py)

    ```python
    @functools.singledispatch
    def foo(a: TTensor, arg1: Type) -> TTensor:
        """@
        __description__

        :param a: The input tensor.
        :param arg1: __description__
        :return: __description__
        """
        if isinstance(a, tensor.Tensor):
            return tensor.Tensor(foo(a.data, axis))
        return NotImplemented(f"Function `foo` is not implemented for {type(a)}")
    ```

3. Add backend specific implementation of method to:

    - [numpy_function.py](numpy_function.py)

        ```python
        @functions.foo.register(np.ndarray)
        @functions.foo.register(np.number)
        def _(a: TType, arg1: Type) -> np.ndarray:
            return np.foo(a, arg1)
        ```

    - [torch_function.py](torch_function.py)

        ```python
        @functions.foo.register(torch.Tensor)
        def _(a: torch.Tensor, arg1: Type) -> torch.Tensor:
            return torch.foo(a, arg1)
        ```

4. Add test of method to [test template](tests/shared/test_templates/template_test_nncf_tensor.py) for nncf.Tensor class


### Add new backend

1. Add backend specific implementation for all function from [function.py](function.py) in `<NEW_BACKEND>_functions.py` file.

2. Add `test_tensor.py` in backend-specific t directory for tests that inherited from class `TemplateTestNNCFTensorOperators`

```python
class TestNPNNCFTensorOperators(TemplateTestNNCFTensorOperators):
    @staticmethod
    def to_tensor(x):
        return np.array(x)  # Function to initialize tensor from list
```

2. Add new backend type to `mock_modules` list in [docs/api/source/conf.py](https://github.com/openvinotoolkit/nncf/blob/develop/docs/api/source/conf.py#L131)

```python
mock_modules = [
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "openvino",
    "tensorflow",
    "tensorflow_addons",
    "nncf.experimental.tensor.torch_functions",
    "nncf.experimental.tensor.numpy_functions",
    "nncf.experimental.tensor.<NEW_BACKEND>_functions",
]
```
