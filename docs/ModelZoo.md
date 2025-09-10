# NNCF Compressed Model Zoo

Ready-to-use **Compressed LLMs** can be found on [OpenVINO Hugging Face page](https://huggingface.co/OpenVINO#models). Each model card includes NNCF parameters that were used to compress the model.

**INT8 Post-Training Quantization** ([PTQ](../README.md#post-training-quantization)) results for public Vision, NLP and GenAI models can be found on [OpenVino Performance Benchmarks page](https://docs.openvino.ai/2025/about-openvino/performance-benchmarks.html). PTQ results for ONNX models are available in the [ONNX](#onnx) section below.

## PyTorch

### PyTorch NLP (HuggingFace Transformers-powered models)

<table>
  <thead>
    <tr>
      <th>PyTorch Model</th>
      <th><img width="20" height="1">Compression algorithm<img width="20" height="1"></th>
      <th>Dataset</th>
      <th>Accuracy (<em>drop</em>) %</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">BERT-base-cased</td>
      <td align="left">• QAT: INT8</td>
      <td>CoNLL2003</td>
      <td>99.18 (-0.01)</td>
    </tr>
    <tr>
      <td align="left">BERT-base-cased</td>
      <td align="left">• QAT: INT8</td>
      <td>MRPC</td>
      <td>84.8 (-0.24)</td>
    </tr>
    <tr>
      <td align="left">BERT-base-chinese</td>
      <td align="left">• QAT: INT8</td>
      <td>XNLI</td>
      <td>77.22 (0.46)</td>
    </tr>
    <tr>
      <td align="left">BERT-large<br>(Whole Word Masking)</td>
      <td align="left">• QAT: INT8</td>
      <td>SQuAD v1.1</td>
      <td>F1: 92.68 (0.53)</td>
    </tr>
    <tr>
      <td align="left">DistilBERT-base</td>
      <td align="left">• QAT: INT8</td>
      <td>SST-2</td>
      <td>90.3 (0.8)</td>
    </tr>
    <tr>
      <td align="left">GPT-2</td>
      <td align="left">• QAT: INT8</td>
      <td>WikiText-2 (raw)</td>
      <td>perplexity: 20.9 (-1.17)</td>
    </tr>
  <tr>
      <td align="left">MobileBERT</td>
      <td align="left">• QAT: INT8</td>
      <td>SQuAD v1.1</td>
      <td>F1: 89.4 (0.58)</td>
    </tr>
    <tr>
      <td align="left">RoBERTa-large</td>
      <td align="left">• QAT: INT8</td>
      <td>MNLI</td>
      <td>matched: 89.25 (1.35)</td>
    </tr>
  </tbody>
</table>

## ONNX

### ONNX Classification

<table>
  <thead>
    <tr>
      <th>ONNX Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>Accuracy (<em>drop</em>) %</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">DenseNet-121</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>60.16 (0.8)</td>
    </tr>
    <tr>
      <td align="left">GoogleNet</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>66.36 (0.3)</td>
    </tr>
    <tr>
      <td align="left">MobileNet V2</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>71.38 (0.49)</td>
    </tr>
    <tr>
      <td align="left">ResNet-50</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>74.63 (0.21)</td>
    </tr>
    <tr>
      <td align="left">ShuffleNet</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>47.25 (0.18)</td>
    </tr>
    <tr>
      <td align="left">SqueezeNet V1.0</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>54.3 (0.54)</td>
    </tr>
    <tr>
      <td align="left">VGG‑16</td>
      <td align="left">PTQ</td>
      <td>ImageNet</td>
      <td>72.02 (0.0)</td>
    </tr>
  </tbody>
</table>

### ONNX Object Detection

<table>
  <thead>
    <tr>
      <th>ONNX Model</th>
      <th>Compression algorithm</th>
      <th>Dataset</th>
      <th>mAP (<em>drop</em>) %</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td align="left">SSD1200</td>
      <td align="left">PTQ</td>
      <td>COCO2017</td>
      <td>20.17 (0.17)</td>
    </tr>
    <tr>
      <td align="left">Tiny-YOLOv2</td>
      <td align="left">PTQ</td>
      <td>VOC12</td>
      <td>29.03 (0.23)</td>
    </tr>
  </tbody>
</table>
