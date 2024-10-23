# Tensors wrapper

The `Tensor` class is a wrapper class that provides a common interface for different types of tensors,
such as NumPy and PyTorch. This allows algorithms to be written that are abstracted from the underlying model type,
making them more portable and reusable.

## Usage

Common algorithms should use wrapped tensors and provide the unwrapped tensor to the backend-specific function.

### Initialization Tensor

```python
from nncf.tensor import Tensor

import numpy as np
numpy_array = np.array([1,2])
nncf_tensor = Tensor(numpy_array)

import torch
torch_tensor = np.array([1,2])
nncf_tensor = Tensor(torch_tensor)
```

### Create Tensor

The function for creating a tensor requires a backend argument to specify the type of data to be created. It supports both numpy array and torch tensor.

```python
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorBackend

# create zeros tensor where data is numpy array
nncf_tensor = fns.zeros((2,2), backend=TensorBackend.numpy)

# create zeros tensor where data is troch tensor
nncf_tensor = fns.zeros((2,2), backend=TensorBackend.torch)
```

### Math operations

All math operations are overridden to operated with wrapped object and return `Tensor`

```python
tensor_a = Tensor(np.array([1,2]))
tensor_b = Tensor(np.array([1,2]))
tensor_a + tensor_b  # Tensor(array([2, 4]))
```

**NOTE** Division operations for the numpy backend are performed with warnings disabled for the same for all backends.

### Comparison operators

All math operations are overridden to operated with wrapped object and return `Tensor`

```python
tensor_a = Tensor(np.array([1,2]))
tensor_b = Tensor(np.array([1,2]))
tensor_a < tensor_b  # Tensor(array([False, False]))
```

### Method of the Tensor class

Some methods of the tensors available from wrapped Tensor class like `max`, `flatten` and other common methods.

```python
nncf_tensor.max()  # Tensor(2)
```

### Functions over Tensor

All available functions you can found in the functions module.

```python
from nncf.tensor import functions as fns
fns.max(nncf_tensor)  # Tensor(2)
```

**NOTE** A function requires at least one positional argument, which is used to dispatch the function
to the appropriate implementation depending on the type of argument.

```python
fns.max(nncf_tensor)  # Correct
fns.max(a=nncf_tensor)  # TypeError: wrapper requires at least 1 positional argument
```

### Loop over Tensor

For `Tensor` available `TensorIterator` that return `Tensor`

```python
tensor_a = Tensor(np.array([1,2]))
for x in tensor_a:
    print(x)

# Tensor(1)
# Tensor(2)
```

### Get element by index Tensor

```python
tensor_a = Tensor(np.array([[1],[2]]))
tensor_a[0]    # Tensor(array([1]))
tensor_a[0:2]  # Tensor(array([[1],[2]]))
```

## Class feature enhancement

**NOTE** Use names and descriptions in numpy style.

### Add new method or function

1. Add method to [class Tensor](tensor.py)

    ```python
    class Tensor:
        ...
        def foo(self, arg1: Type) -> "Tensor":
            return fns.foo(self, arg1)
    ```

2. Add function to functions module

    ```python
    @functools.singledispatch
    def foo(a: TTensor, arg1: Type) -> TTensor:
        """
        __description__

        :param a: The input tensor.
        :param arg1: __description__
        :return: __description__
        """
        if isinstance(a, tensor.Tensor):
            return tensor.Tensor(foo(a.data, axis))
        return NotImplemented(f"Function `foo` is not implemented for {type(a)}")
    ```

    **NOTE** For the case when the first argument has type `List[Tensor]`, use the `_dispatch_list` function. This function dispatches function by first element in the first argument.

    ```python
    @functools.singledispatch
    def foo(x: List[Tensor], axis: int = 0) -> Tensor:
        if isinstance(x, List):
            unwrapped_x = [i.data for i in x]
            return Tensor(_dispatch_list(foo, unwrapped_x, axis=axis))
        raise NotImplementedError(f"Function `foo` is not implemented for {type(x)}")
    ```

3. Add backend specific implementation of method to corresponding module:

    - `functions/numpy_*.py`

        ```python
        @_register_numpy_types(fns.foo)
        def _(a: TType, arg1: Type) -> np.ndarray:
            return np.foo(a, arg1)
        ```

    - `functions/torch_*.py`

        ```python
        @fns.foo.register(torch.Tensor)
        def _(a: torch.Tensor, arg1: Type) -> torch.Tensor:
            return torch.foo(a, arg1)
        ```

4. Add test of method to [test template](/tests/cross_fw/test_templates/template_test_nncf_tensor.py) for Tensor class

### Add new backend

1. Add backend specific implementation for all function from functions module in `functions/<NEW_BACKEND>_*.py` file.

2. Add `test_tensor.py` in backend-specific directory for tests that inherited from class `TemplateTestNNCFTensorOperators`

    ```python
    class TestNPNNCFTensorOperators(TemplateTestNNCFTensorOperators):
        @staticmethod
        def to_tensor(x):
            return np.array(x)  # Function to initialize tensor from list
    ```

3. Add new backend type to `mock_modules` list in [docs/api/source/conf.py](/docs/api/source/conf.py#L131)

    ```python
    mock_modules = [
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime",
        "openvino",
        "tensorflow",
        "nncf.tensor.functions.torch_*",
        "nncf.tensor.functions.numpy_*",
        "nncf.tensor.functions.<NEW_BACKEND>_*",
    ]
    ```
