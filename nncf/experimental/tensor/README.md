# Tensors wrapper

The `Tensor` class is a wrapper class that provides a common interface for different types of tensors,
such as NumPy and PyTorch. This allows algorithms to be written that are abstracted from the underlying model type,
making them more portable and reusable.

## Usage

Common algorithms should use wrapped tensors and provide the unwrapped tensor to the backend-specific function.

### Initialization Tensor

```python
from nncf.experimental.tensor import Tensor

import numpy as np
numpy_array = np.array([1,2])
nncf_tensor = Tensor(numpy_array)

import torch
torch_tensor = np.array([1,2])
nncf_tensor = Tensor(torch_tensor)
```

### Math operations

All math operations are overrided to operated with wrapped object and return `Tensor`

```python
tensor_a = Tensor(np.array([1,2]))
tenor_b = Tensor(np.array([1,2]))
tensor_a + tenor_b  # Tensor(array([2, 4]))
```

### Comparison operators

All math operations are overrided to operated with wrapped object and return `Tensor`

```python
tensor_a = Tensor(np.array([1,2]))
tenor_b = Tensor(np.array([1,2]))
tensor_a < tenor_b  # Tensor(array([False, False]))
```

### Method of the Tensor class

Some methods of the tensors available from wrapped Tensor class like `max`, `flatten` and other common methods.

```python
nncf_tensor.max()  # Tensor(2)
```

### Functions over Tensor

All available functions you can found in [functions.py](functions.py).

```python
from nncf.experimental.tensor import functions as fns
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

2. Add function to [function.py](function.py)

    ```python
    @tensor_dispatch()
    def foo(a: TTensor, arg1: Type) -> TTensor:
        """
        __description__

        :param a: The input tensor.
        :param arg1: __description__
        :return: __description__
        """
    ```

    **NOTE** To control work with Tensors, different types of wrapper functions can be selected
    `@tensor_dispatch(wrapper_type=WrapperType.TensorToTensor)`:

    - `WrapperType.TensorToTensor` (default) expects Tensor as first argument, result will be wrapped to Tensor.
    - `WrapperType.TensorToAny` expects Tensor as first argument, result will not be wrapped to Tensor.
    - `WrapperType.TensorToList` expects Tensor as first argument, each element in result list will be wrapped to Tensor.
    - `WrapperType.ListToTensor` expects List of Tensors as first argument, result will be wrapped to Tensor.

3. Add backend specific implementation of method to:

    - [numpy_function.py](numpy_functions.py)

        ```python
        @_register_numpy_types(fns.foo)
        def _(a: TType, arg1: Type) -> np.ndarray:
            return np.foo(a, arg1)
        ```

    - [torch_function.py](torch_functions.py)

        ```python
        @fns.foo.register(torch.Tensor)
        def _(a: torch.Tensor, arg1: Type) -> torch.Tensor:
            return torch.foo(a, arg1)
        ```

4. Add test of method to [test template](../../../tests/shared/test_templates/template_test_nncf_tensor.py) for Tensor class

### Add new backend

1. Add backend specific implementation for all function from [function.py](function.py) in `<NEW_BACKEND>_functions.py` file.

2. Add `test_tensor.py` in backend-specific t directory for tests that inherited from class `TemplateTestNNCFTensorOperators`

    ```python
    class TestNPNNCFTensorOperators(TemplateTestNNCFTensorOperators):
        @staticmethod
        def to_tensor(x):
            return np.array(x)  # Function to initialize tensor from list
    ```

3. Add new backend type to `mock_modules` list in [docs/api/source/conf.py](https://github.com/openvinotoolkit/nncf/blob/develop/docs/api/source/conf.py#L131)

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
