# Testing Results

This document captures the performance and inference time results of various optimization and decoding methods tested on different GPT-2 model sizes.

## Combined Optimization Methods Testing

Results for `combined_optimisation_methods_testing.py` on a MacBook Pro 16-inch 2019:

Note: BT - BetterTransformers a Hugging Face feature for accelerated inference.

### GPT2-SMALL (10 runs)

| Optimization Technique                | Inference Time (s) |
|---------------------------------------|-------------------:|
| FP32                                  |              0.53  |
| BT                                    |              0.51  |
| Quantized Dynamic                     |              0.42  |
| Quantized Dynamic with Multithreading |              0.28  |
| Multithreading                        |              0.34  |

### GPT2-MEDIUM (10 runs)

| Optimization Technique                | Inference Time (s) |
|---------------------------------------|-------------------:|
| FP32                                  |              1.30  |
| BT                                    |              1.17  |
| Quantized Dynamic                     |              1.19  |
| Quantized Dynamic with Multithreading |              0.83  |
| Multithreading                        |              0.92  |

### GP2-LARGE (10 runs)

| Optimization Technique                | Inference Time (s) |
|---------------------------------------|-------------------:|
| FP32                                  |              2.55  |
| BT                                    |              2.53  |
| Quantized Dynamic                     |              2.49  |
| Quantized Dynamic with Multithreading |              1.92  |
| Multithreading                        |              1.94  |

All these optimization methods are based on the assumption of multiple simultaneous processes, which can leverage proper data distribution and processing. As a result, the same speed gains may not be observed in a single inference scenario, such as with a user interface.

### GPT2-MEDIUM (1 run)

| Optimization Technique                | Inference Time (s) |
|---------------------------------------|-------------------:|
| FP32                                  |              1.27  |
| BT                                    |              1.28  |
| Quantized Dynamic                     |              1.19  |
| Quantized Dynamic with Multithreading |              1.43  |
| Multithreading                        |              1.49  |

When generating a single output, the Quantized Dynamic Optimization alone provided the fastest inference time.

## Decoder Method Testing

Results for `decoder_method_testing.py`:

### GPT2

| Decoding Method         | Average Inference Time (s) |
|-------------------------|---------------------------:|
| Greedy Search           |                      0.79  |
| Contrastive Search      |                      1.19  |
| Sampling                |                      0.81  |
| Diverse Beam Search     |                      1.26  |
| Top P Nucleus Sampling  |                      0.91  |

### GPT2-MEDIUM

| Decoding Method         | Average Inference Time (s) |
|-------------------------|---------------------------:|
| Greedy Search           |                      2.02  |
| Contrastive Search      |                      3.20  |
| Sampling                |                      2.06  |
| Diverse Beam Search     |                      3.21  |
| Top P Nucleus Sampling  |                      2.20  |

### GPT2-LARGE

| Decoding Method         | Average Inference Time (s) |
|-------------------------|---------------------------:|
| Greedy Search           |                      3.95  |
| Contrastive Search      |                      6.36  |
| Sampling                |                      4.09  |
| Diverse Beam Search     |                      6.36  |
| Top P Nucleus Sampling  |                      4.35  |

Qualitatively, sampling methods tend to generate more coherent text with a reduced likelihood of repetition. The GPT-2 Medium model offers a good balance between speed and output quality. As model size increases, inference time also increases without a corresponding significant improvement in output quality.

## Optimization Testing

Results for `optimisation_testing.py`:

Note: TC is torch.compile(), but was omitted after the first test because it is not compatible with Python 12.2+, and the speed was not sufficient enough to cater to it.

### GPT2-SMALL

| Optimization Technique                | Inference Time (s) |
|---------------------------------------|-------------------:|
| FP32                                  |              0.48  |
| TC (omitted)                          |              0.48  |
| BT                                    |              0.47  |
| Quantized Dynamic                     |              0.39  |
| Quantized Dynamic with Multithreading |              0.27  |
| Multithreading                        |              0.32  |

### GPT2-MEDIUM

| Optimization Technique                | Inference Time (s) |
|---------------------------------------|-------------------:|
| FP32                                  |              1.32  |
| BT                                    |              1.26  |
| Quantized Dynamic                     |              1.52  |
| Quantized Dynamic with Multithreading |              0.82  |
| Multithreading                        |              0.92  |

### GP2-LARGE

| Optimization Technique                | Inference Time (s) |
|---------------------------------------|-------------------:|
| FP32                                  |              2.85  |
| BT                                    |              2.87  |
| Quantized Dynamic                     |              2.36  |
| Quantized Dynamic with Multithreading |              1.95  |
| Multithreading                        |              1.98  |

The optimization methods are designed for scenarios with multiple processes, which may not reflect the same speed benefits in a single-inference user interface setup. The use of GPT-XL was not considered as its long inference time falls outside the scope of efficient instance creation for this task.
