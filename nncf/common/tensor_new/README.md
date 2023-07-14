# How to add new methods to class Tensor

1. Add method to [class Tensor](tensor.py)
2. Add backend specific implementation of method to:
    - [numpy_ops.py](numpy_ops.py)
    - [torch_ops.py](/nncf/torch/torch_ops.py)
3. Add test of method to [test template](tests/shared/test_templates/template_test_nncf_tensor.py)


# How to add new function to math module

1. Add function to [math module](math.py):
  - name of function and arguments should be in numpy style
  - should contain arguments that exists in all backends
  - docstring
2. Add backend specific implementation of function to:
    - [numpy_ops.py](numpy_ops.py)
    - [torch_ops.py](/nncf/torch/torch_ops.py)
3. Add test of method to [test template](tests/shared/test_templates/template_test_nncf_tensor.py)
