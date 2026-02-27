# NNCF Tools

## Visualization of Weight Compression results

The [visualize_compression_results.py](visualize_compression_results.py) script is a useful tool for visualizing the results of weight compression.
The result of the script is a .md file with a table:

| mode                                      | %int4   | %int8   | lora<br>rank   | average<br>relative<br>error   | compression<br>rate   |
|:------------------------------------------|:--------|:--------|:---------------|:-------------------------------|:----------------------|
| fp32                                      | 0%      | 0%      |                | 0.0%                           | 1.0x                  |
| int8                                      | 0%      | 100%    |                | 1.0%                           | 4.0x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 256.0          | 3.9%                           | 6.0x                  |
| int4 + scale estimation                   | 40%     | 60%     |                | 4.1%                           | 4.8x                  |
| int4 + scale estimation                   | 60%     | 40%     |                | 4.3%                           | 5.4x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 128.0          | 4.6%                           | 6.5x                  |
| int4 + scale estimation                   | 80%     | 20%     |                | 5.7%                           | 6.1x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 8.0            | 5.8%                           | 7.1x                  |
| int4 + scale estimation + gptq            | 100%    | 0%      |                | 6.1%                           | 7.1x                  |
| int4 + scale estimation                   | 100%    | 0%      |                | 7.5%                           | 7.1x                  |
| int4                                      | 100%    | 0%      |                | 11.9%                          | 7.1x                  |

Also it plots a trade-off between accuracy and footprint by processing a CSV file in a specific format.
The resulting images are employed for [the relevant section](/docs/usage/post_training_compression/weights_compression/Usage.md#accuracyfootprint-trade-off) in the Weight Compression documentation:

![alt text](/docs/usage/post_training_compression/weights_compression/phi3_asym.png)

### CSV-file format

The input file should contain the following columns:

- `mode` - The string indicating the compression method used for the model. The 'fp32' mode corresponds to the uncompressed version. To calculate the accuracy-footprint trade-off, the following words must be present in at least one row: "gptq", "int4", "fp32", "int8".
- `%int4` - The ratio of int4 layers.
- `%int8` - The ratio of int8 layers.
- `lora rank` - The rank of the adapters used in Lora Correction algorithm.
- `plot name` - Short names for annotation in the plot.
- `model size, Gb` - The size of the corresponding model in Gb.
- `wikitext, word perplexity` - Word perplexity on the Wikitext dataset, measured using rolling loglikelihoods in the [lm_eval tool](https://github.com/EleutherAI/lm-evaluation-harness).
- `lambada-openai, acc` - Accuracy on the Lambada-OpenAI dataset, measured using [lm_eval tool](https://github.com/EleutherAI/lm-evaluation-harness).
- `lambada-openai, perplexity` - Perplexity on the Lambada-OpenAI dataset, measured using the [lm_eval tool](https://github.com/EleutherAI/lm-evaluation-harness).
- `WWB, similarity` - Similarity, measured using the [WWB tool](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench).

### Example of script usage

```shell
python visualize_compression_results.py --input-file data/llama2_asym.csv --output-dir output_dir
```
