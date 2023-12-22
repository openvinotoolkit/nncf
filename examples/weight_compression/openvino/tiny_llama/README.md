# Weight compression of TinyLLama OpenVINO Model

This example demonstrates how to find optimal ratio and group size to compress weights of the TinyLLama model from the HuggingFace Transformers framework. OpenVINO backend supports mixed precision weight compression with a 4-bit data type as a primary precision. The fastest mixed-precision mode is `INT4_SYM`, but it may lead to a significant accuracy degradation. In this example, the allowed maximum deviation from the original model is `0.2`. If the similarity of the compressed model is not satisfying, there are 2 hyper-parameters to tune: `group_size` and `ratio`. Lower group size and less ratio of 4-bit layers usually improve accuracy at the sacrifice of inference speed.

The example includes the following steps:

- Download and prepare `wikitext` dataset.
- Download and export `PY007/TinyLlama-1.1B-step-50K-105b` model to OpenVINO IR.
- Compress weights of the model with NNCF Weight compression algorithm.
- Find optimal ratio and group size if acceptable similarity is not achieved.
- Save model with the best hyperparameters found.
- Measure similarity and footprint of the compressed model.

## Install requirements

At this point it is assumed that you have already installed NNCF. You can find information on installation NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:

```bash
pip install -r requirements.txt
```

## Run Example

The example is fully automated. Just run the following comman in the prepared Python environment:

```bash
python main.py
```

## See also

- [Weight compression](../../../../docs/compression_algorithms/CompressWeights.md)
