
### Block Floating Point Hardware Modelling
Some AI hardware accelerators use custom floating point arithmetic to achieve higher performance. Computations using custom arithmetic usually lead to different results from IEEE-compliant FP32 or FP16 arithmetic. These differences, if left unhandled, may lead to lower accuracy when evaluating deep learning networks. One way to handle custom arithmetic is with [Quantization](Quantization.md). Another way is to model the custom arithmetic in training framework such as NNCF, and run a few re-training epochs to fine-tune the weights. This document describes addition of block floating point arithmetic to NNCF.

#### BFP Format Description
[Block floating point](https://en.wikipedia.org/wiki/Block_floating_point) performs floating point operations by:
1. Grouping nearby floating point values into fixed-size blocks.
1. Choosing a common exponent for values within each block and re-aligning mantissas to match the new exponent.
1. Performing integer arithmetic on mantissas of the blocked values.
1. Converting result back to standard floating point format.

Step 3 above (performing integer arithmetic) usually consumes vast majority of hardware resources, whereas all other steps can either be done once on host CPU attached to hardware accelerator (e.g. blocking of weights for a network only needs to be done once), or consume relatively small amount of hardware. Using block floating point instead of regular floating point uses less hardware resources but may cause accuracy degration due to effect of blocking.

Block floating point is usually combined with smaller mantissa sizes. For example, **int5bfp** block floating point format has:
- 5 integer bits (including 1 sign bit)
- 5 exponent bits
- block size 32
Without blocking, this would be comparable to FP9 floating point format: 1 sign bit, 1 implicit mantissa bit, 3 explicit mantissa bits, 5 exponent bits.

**INSERT NICE IMAGES HERE**

Note that BFP computation need not be symmetric. For example, network activations could be represented as **int5bfp** but network weights as **int4bfp**. Selection of appropriate BFP format depends on what the hardware supports and achieved accuracy of network of interest with selected precisions.

#### Example Hardware Support for BFP
Major compute blocks: 
- data storage in external memory
- data storage on-chip
- compute engine 
- auxiliary compute engine (fixed or floating point)
Need for precision conversions between host, accelerator hw, and between different hardware blocks

#### BFP Handling in NNCF
To successfully train to BFP-enabled hardware, NNCF must model as close as possible target hardware's arithmetic. Current implementation models hardware as implemented in Intel FPGA Deep Learning Accelerator Suite, supported by the Intel OpenVINO™ toolkit. 

NNCF inserts a Quantization layer for each input of every convolution layer. This quantization layer performs converts to lower precision and blocks FP32 activations/weights, and then converts them back to FP32. The exact conversion mechanism is described in hardware models. Such models include number of integer and exponent bits, block size, and also exact rounding methods.
