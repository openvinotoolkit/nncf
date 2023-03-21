# Example of estimate activation sparsity statistics

This example demonstrates the api for collecting and storing activation sparsity statistics.
Activation sparsity is the average percentage of zeros in the incoming tensor on specified port id.


## Usage

```python
python main.py -m <path_to_xml> -d <data_path>

# For all nodes
python main.py -m <path_to_xml> -d <data_path> -t all

# For Convolution and MatMul nodes
python main.py -m <path_to_xml> -d <data_path> -t Convolution,MatMul
```


## Output format
Output dictionary collect name of layer, port id and sparsity level.

```
{
    "/nncf_module/layer1/layer1.0/conv2/Conv/WithoutBiases": [
        {
            "port_id": 0,
            "sparsity_level": 0.41130510602678577
        }
    ],
    "/nncf_module/layer1/layer1.1/conv1/Conv/WithoutBiases": [
        {
            "port_id": 0,
            "sparsity_level": 0.40485699089205995
        }
    ]
}
```
