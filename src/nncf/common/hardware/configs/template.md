```json5
{
    /* Supported compression algorithm configurations */
    "config": {
        /* Quantization algorithm configurations */
        "quantization": {
            "q8": { // Alias name for set of hyperparameters
                "bits": 8, // Number of quantization bits
                "mode": [  // Quantization mode in order of priority, the first mode has higher priority
                    "symmetric",
                    "asymmetric"
                ],
                /*
                 * Granularity: one scale for output tensor or a scale for each output channel.
                 * values are listed in order of priority.
                 */
                "granularity": [
                    "pertensor",
                    "perchannel"
                ],
                /*
                 * Narrow range: should NNCF use 2**num_bits quants or 2**num_bits - 1
                 */
                "narrow_range": false
            },
            "q8_sym_tnr_signed": { // Alias name for set of hyperparameters
                "bits": 8, // Number of quantization bits
                "mode": "symmetric", // Quantization mode
                "granularity": "pertensor", // Granularity: one scale for output tensor
                "signedness_to_force": true, // signedness_to_force: True if the quantizer must be signed, False if must be unsigned,
                                             // None if the signed/unsigned attribute should be determined based on the incoming activation
                                             // statistics during range initialization.
                "narrow_range": true
            }
        }
    },
    /* Supported operations */
    "operations": [
        {
            "type": "Convolution", // Operation type
            "quantization": { // Compression algorithm type
                /*
                 * List of supported configurations of quantization algorithms
                 * for activations in order of priority, the first configuration has higher priority.
                 */
                "activations": [
                    {"bits": 2, "mode": ["symmetric"], "granularity":["pertensor"]},
                    {"bits": 4, "mode": ["symmetric"], "granularity":["pertensor"]},
                    {"bits": 8, "mode": ["symmetric", "asymmetric"], "granularity":["pertensor", "perchannel"]}
                ],
                /*
                 * List of supported configurations of quantization algorithms
                 * for weights in order of priority, the first configuration has higher priority.
                 */
                "weights":[
                    {"bits": 4 ,"mode": ["symmetric"], "granularity":["pertensor"]},
                    {"bits": 8, "mode": ["symmetric", "asymmetric"], "granularity":["pertensor", "perchannel"]}
                ]
            }
        },
        {
            "type": "MatMul", // Operation type
            "quantization": { // Compression algorithm type
                 /*
                 * List of supported configurations of quantization algorithms
                 * for activations in order of priority, the first configuration has higher priority.
                 */
                "activations": ["q8"],
                /*
                 * List of supported configurations of quantization algorithms
                 * for weights in order of priority, the first configuration has higher priority.
                 */
                "weights": ["q8_sym_tnr_-127_127"]
            }
        }
    ]
}
```
