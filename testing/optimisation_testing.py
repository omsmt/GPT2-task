import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import torch
import numpy as np
import copy 
import concurrent.futures
import sys

# Check Python version
python_version = sys.version_info

# Initialize the model and tokenizer
model_name = "gpt2"  # Using "gpt2-medium" for demonstration; adjust model size as needed.
set_seed(42)  # For reproducibility
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).eval()  # Model in eval mode
pad_token_id = tokenizer.eos_token_id
input_text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# Function to perform warmup and measure inference time
def measure_inference_time(model, input_ids, attention_mask, pad_token_id, num_trials=10):
    with torch.no_grad():  # Disable gradient calculation for inference
        # Warm-up
        model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=25)
        # Measure inference time
        start_time = time.time()
        for _ in range(num_trials):
            model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=25, do_sample=True, top_p=0.95, top_k=60)
        elapsed_time = (time.time() - start_time) / num_trials
    return elapsed_time

# Function to perform model inference
def model_inference(model, input_ids, attention_mask, pad_token_id):
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=25)
    return outputs

# Function to measure inference time with multithreading
def measure_inference_time_multithreaded(model, input_ids, attention_mask, pad_token_id, num_trials=10, num_workers=6):
    torch.set_num_threads(1)  # Limit the number of threads per model to prevent oversubscription
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(model_inference, model, input_ids, attention_mask, pad_token_id)
            for _ in range(num_trials)
        ]
        concurrent.futures.wait(futures)
    elapsed_time = (time.time() - start_time) / num_trials
    return elapsed_time

# Torch Compile
def apply_torch_compile(model):
    model_tc = torch.compile(model)
    # model = torch.jit.script(model, (input_ids, attention_mask))
    return model_tc

# Better Transformers
def apply_better_transformers(model):
    model_bt = copy.deepcopy(model)
    model_bt.to_bettertransformer()
    return model_bt

# Dynamic Quantization
def apply_dynamic_quantization(model):
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model_quantized

# Testing different optimizations
def test_optimizations():
    results = {}

    # Default configuration (FP32)
    time_fp32 = measure_inference_time(model, input_ids, attention_mask, pad_token_id)
    results["FP32"] = time_fp32

    if python_version < (3, 12):
        # Torch Compile 
        model_tc = apply_torch_compile(model) # not compatible with python 3.12+
        time_bt = measure_inference_time(model_tc, input_ids, attention_mask, pad_token_id)
        results["TC"] = time_bt

    # Better Transformers 
    model_bt = apply_better_transformers(model)
    time_bt = measure_inference_time(model_bt, input_ids, attention_mask, pad_token_id)
    results["BT"] = time_bt

    # Dynamic Quantization
    model_quantized_dynamic = apply_dynamic_quantization(model)
    time_quantized_dynamic = measure_inference_time(model_quantized_dynamic, input_ids, attention_mask, pad_token_id)
    results["Quantized Dynamic"] = time_quantized_dynamic

    # Dynamic Quantization with multithreading
    model_quantized_dynamic = apply_dynamic_quantization(model)
    time_quantized_dynamic_mt = measure_inference_time_multithreaded(model_quantized_dynamic, input_ids, attention_mask, pad_token_id)
    results["Quantized Dynamic with Multithreading"] = time_quantized_dynamic_mt

    # Multithreading
    time_quantized_dynamic_mt = measure_inference_time_multithreaded(model, input_ids, attention_mask, pad_token_id)
    results["Multithreading"] = time_quantized_dynamic_mt
    return results

# Execute the tests
optimization_results = test_optimizations()

# Display the results
print("Inference Time (seconds) per Optimization Technique:")
for technique, time in optimization_results.items():
    print(f"{technique}: {time:.4f}s")
