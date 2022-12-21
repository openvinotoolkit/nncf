Install the packages needed for samples by running the following in the current directory:

```
pip install -r requirements.txt
```

One of the needed package - torchvision.
The minor version of torchvision should always match the minor version of installed torch package.
torchvision minor version is torch minor version +1. For instance, torch 1.8.2 goes with torchvision 0.9.2.

By default, if there is no torchvision in your Python environment it installs the package that is compatible with 
the best known torch version (`BKC_TORCH_VERSION` in the code). In that case if your environment has the torch version, 
which is different from best known one, you should install the corresponding torchvision package by yourself.

For example, if you need torch 1.9.1 (not best known version) with CUDA11 support, we recommend specifying the 
corresponding torchvision version as follows in the root nncf directory: 

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install .[torch]
pip install -r examples/torch/requirements.txt
```
