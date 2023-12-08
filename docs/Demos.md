## Demos, Tutorials and Samples

Where to locate the below subsections?
- Either keep all subsections in the main readme
- Or move all subsections to a separate docs file, e.g. `docs/Demos.md`
- Or keep only a single subsection, e.g. Jupyter Tutorials and move the rest to a separate file

### Jupyter Notebook Tutorials
<Here we present basic jupyter notebooks which contain compression methods. These are simple and short, notebooks IDs are generally in range 1** and 3**, hence we refer to them as tutorials\>

|                                                                                                                                                                                                 Tutorial                                                                                                                                                                                                  | Compression Algorithm | Backend | Domain |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| :---: | :---:  | :---: |
| [BERT Quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/105-language-quantize-bert/105-language-quantize-bert.ipynb)                                           | Post-Training Quantization                       | OpenVINO   | NLP                                |
| [MONAI Segmentation Model Quantization](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/110-ct-segmentation-quantize)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F110-ct-segmentation-quantize%2F110-ct-scan-live-inference.ipynb)                                             | Post-Training Quantization                       | OpenVINO   | Segmentation                       |
| [PyTorch Model Quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/112-pytorch-post-training-quantization-nncf)                                                                                                                                                                                                                                                      | Post-Training Quantization                       | PyTorch    | Image Classification               |
| [TensorFlow Model Quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)                                                                                                                                                                                                                                                              | Post-Training Quantization                       | Tensorflow | Image Classification               |
| [Migrating from POT to NNCF Quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/111-yolov5-quantization-migration)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/111-yolov5-quantization-migration/111-yolov5-quantization-migration.ipynb) | Post-Training Quantization                       | OpenVINO   | Object detection                   |
| [Quantization with Accuracy Control](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/122-quantizing-model-with-accuracy-control)                                                                                                                                                                                                                                               | Post-Training Quantization with Accuracy Control | OpenVINO   | Speech-to-Text<br>Object Detection |
| [TensorFlow Training-Time Compression](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)                                                                                                                                                                                                                                                       | Training-Time Compression                        | Tensorflow | Image Classification               |



### Jupyter Notebook Demos
<Here we present jupyter notebooks with real-life models which contain some kind of NNCF compression as a part of them. These are generally more complex and education is not the sole purpose, more of showing-off. Hence they are referred to as Demos here. \>

### Post-Training Quantization Samples
<Here will be the PTQ examples from `examples/post_training_quantization`\>

### Training-Time Compression Samples
<Here will be the TTC examples from `examples/torch` and `examples/tensorflow`\>