# Mia's Technical Task

Mia's Technical Task is an application that integrates a GPT-2 model for generating text based on user input. It provides a user-friendly interface for loading the model, entering prompts, and displaying the generated text output. This project is built using Python, with dependencies on libraries such as `torch`, `transformers`, `imgui`, and `glfw` for the graphical user interface and model handling. The code is heavily reliant upon the HuggingFace transfomers library, in particular GPT2 libraries which can be explored in [Reference 1](#reference-1). 
There is also a selection of scripts for the investigation of CPU inference optimisation. These can be found in the `testing/` directory.

## Features

- Load GPT-2 model with dynamic quantisation for efficient inference
- User interface for entering prompts and displaying generated text
- Asynchronous text generation to keep the UI responsive
- Customizable font settings for the UI
- Users have the option to select the model size and choose quantization directly through the UI.

## Requirements

- Python 3.8 or later
- `torch`
- `transformers`
- `optimum`
- `imgui`
- `glfw`
- `PyOpenGL`
- `PyOpenGL-accelerate`
- `onnx`
- `onnxruntime`

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/omsmt/GPT2_Task.git
cd GPT2_Task
```

2. **Install dependencies:**

It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

To run the application, execute the main script:

```bash
python3 main.py
```

Upon launching, you will see a window with instructions to load the model and enter your prompt. Follow the UI instructions for text generation.

### UI Guide

- **Model Selection:** Use the dropdown menu to select the size of the GPT-2 model you wish to use. Options include 'gpt2' (the smallest model), 'gpt2-medium', and 'gpt2-large'. Each model size offers a balance between computational efficiency and the ability to capture the intricacies of language patterns, with larger models typically providing more accurate but computationally intensive outputs.
- **Quantisation Option**: To further customize the application's performance and resource utilization, users can opt for quantization. This feature allows for:
  - **Enabled**: Model weights are quantised, reducing the model's size and speeding up inference at the cost of a slight decrease in accuracy.
  - **Disabled**: Full precision weights are used, maximizing accuracy but requiring more storage and computational resources.
- **Load Model:** After selecting your desired model size, click the "Load Model and Weights" button to load the GPT-2 model. This process applies dynamic quantization to the selected model to optimize its size and inference speed without significantly sacrificing output quality.
- **Enter Prompt:** Type your prompt in the input field and press Enter or click the "Generate" button. The model uses a sampling decoder with `max_new_tokens=25` to generate text based on your prompt. This setting limits the computation time and ensures timely responses while still allowing for creative and contextually relevant continuations.
- **View Output:** The generated text and the model inference time will be displayed below. Depending on the complexity of the prompt and the selected model size, the "accuracy" of the generated text—how well it matches or continues from the prompt—can vary. The inference time provides insight into the computational efficiency of each model size under the constraints of dynamic quantization and the sampling decoder.

These enhancements aim to provide a user-friendly interface for exploring the capabilities of GPT-2 models while balancing performance and output quality. Users are encouraged to experiment with different model sizes and prompts to experience the trade-offs between speed and text generation accuracy firsthand.


## Troubleshooting

- Ensure all dependencies are correctly installed and your Python version is compatible.
- If the model fails to load, check the console for any error messages that might indicate the issue.
- For UI-related issues, ensure your system supports OpenGL and has the necessary drivers installed.
- Pray to the computer god's for a miracle

## Testing

This section contains information about additional scripts located in the testing folder for evaluating different aspects of the model and optimisation techniques. Detailed results and discussions are available in the [Testing Results](testing/TESTING_RESULTS.md) document.

**Decoder Method Testing (`decoder_method_testing.py`)**

This script evaluates the performance of different decoding methods (e.g., Greedy Search, Sampling) and model sizes (GPT-2 Small, Medium, Large) in terms of inference time and output quality. It utilizes the transformers library for the GPT-2 model and various decoding strategies.

Main Findings:
- Sampling methods tend to produce more coherent text with a lower likelihood of repetition.
- GPT-2 Medium strikes a good balance between inference speed and output quality.
- Larger models increase inference time without necessarily improving output quality significantly.

**Optimisation Testing (`optimisation_testing.py`)**

This script tests various optimisation techniques (e.g., Dynamic Quantisation, Multithreading) to improve inference speed while maintaining or enhancing output quality. It uses the GPT-2 model from the transformers library and applies optimisations available in PyTorch.

Main Findings:
- Dynamic quantisation significantly reduces inference time, especially when combined with Multithreading.
- Optimisations like Torch Compile (TC) and Better Transformers (BT) offer marginal improvements in inference speed.
- The choice of optimisation technique depends on the specific requirements of the application, such as the balance between speed and accuracy.

**Combining Decoder and Optimisation Methods (`combined_optimisation_methods_testing.py`)**

This script is the same as optimisation_testing.py except the decoder method has been set as Sampling. This allows us to see the inference times for the combination of the chosen decoder method and the optimisation methods.
For a comprehensive analysis of the tests and their results, please consult the scripts and documentation in the testing folder.

**Optimization and Sampling Test (`opt_test_script.py`)**  

This script evaluates the GPT-2 model's performance using dynamic quantization and sampling methods to fine-tune the balance between generation speed and text quality for a specific task. Utilizing a fixed prompt ensures consistent comparisons of output quality and inference times across trials.

Key Strategies:
- **Dynamic Quantization** compresses the model to decrease inference time without drastically impacting text quality, optimizing efficiency.
- **Sampling Methods** introduce variability and creativity in the text generation process, demonstrating the impact of different strategies on output coherence and uniqueness.
- **Fixed Prompt** provides a uniform basis for testing, allowing for reliable analysis of the optimizations' effects.

Despite the potential speed gains from multithreading, it was excluded from the final UI implementation, as the UI generates single outputs at a time. This script illustrates effectiveness of the chosen optimal setup, which unites rapid response times with reasonable quality text generation.

**Bonus: ONNX Model Testing**

A bonus script for initial testing using an ONNX model is included in the `testing/bonus` directory. This script, `onnx_method_testing.py`, provides a quick test environment to perform inference with a GPT-2 ONNX model on a CPU. The ONNX model offers potential optimisation for deployment scenarios where inference speed is a critical factor. The script is adapted from the approach detailed in [Reference 8](#reference-8).

## Future Considerations

With additional time and resources, the following optimisation avenues could be explored:

- Expanding the utilization of ONNX in conjunction with quantisation to assess their suitability and impact on this task. This approach has the potential to allow the use of more computationally intensive models without sacrificing efficiency, potentially leading to improvements in both speed and accuracy.
- Investigating a broader range of decoding strategies and conducting a detailed analysis of their performance metrics to identify the optimal configuration.
- Extending the scope of performance evaluations by increasing the sample size beyond the current set of 10 outputs, thereby enhancing the reliability of our average inference time measurements.

These steps are aimed at enhancing the model's performance and the overall effectiveness of the text generation in the pursuit of achieving the task: the fastest possible autoregressive decoding on a CPU.

## References

For more information on GPT-2 and the implementation of various optimisation and decoding strategies, all of which I found particularly fruitful, the following resources are recommended:


1. [Transformers Library GPT-2 Overview](https://huggingface.co/docs/transformers/v4.38.2/en/model_doc/gpt2#overview) - An overview of the GPT-2 model in the Transformers library documentation.
2. [Setup GPT-2 on Your PC - Medium Article](https://medium.com/codex/setup-gpt2-on-your-pc-6fb7d745355c) - A guide on setting up GPT-2 on your personal computer, including installation steps and basic usage.
3. [Decoding Strategies in Transformers](https://huggingface.co/docs/transformers/generation_strategies#decoding-strategies) - Documentation on various text generation decoding strategies available in the Transformers library.
4. [Torch Compile Tutorial - PyTorch](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) - A tutorial on using `torch.compile` for optimizing PyTorch models for better performance.
5. [Performance Tuning Checklist - PyTorch Serve](https://pytorch.org/serve/performance_checklist.html) - A checklist for optimizing model serving performance in PyTorch.
6. [PyTorch CPU Inference Speed Up - SoftwareG Blog](https://softwareg.com.au/blogs/computer-hardware/pytorch-cpu-inference-speed-up) - Insights into speeding up PyTorch model inference on CPU.
7. [Converting Models to BetterTransformer - Hugging Face Optimum](https://huggingface.co/docs/optimum/bettertransformer/tutorials/convert) - A tutorial on converting transformer models to use the BetterTransformer library for optimized inference.
8. [Inference GPT-2 with OnnxRuntime on CPU - GitHub Microsoft ONNXRuntime](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2_with_OnnxRuntime_on_CPU.ipynb) - A notebook demonstrating how to perform inference with GPT-2 using OnnxRuntime on a CPU.
9. [Usage guide for pyimgui](https://pyimgui.readthedocs.io/en/latest/guide/index.html) - Documentation and guides for pyimgui library.
