import concurrent.futures
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import time
set_seed(15)

def apply_dynamic_quantization(model):
    """Apply dynamic quantization to the model."""
    model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model_quantized

def generate_text(model, tokenizer, prompt, max_length=25):
    """Generate text based on the prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_new_tokens=25, do_sample=True, top_p=0.95, top_k=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def parallel_text_generation(model, tokenizer, prompt, num_runs=10):
    """Run model inference in parallel across multiple threads."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_text, model, tokenizer, prompt) for _ in range(num_runs)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

# Initialize the model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).eval()  # Model in eval mode

# Apply dynamic quantization
model_quantized = apply_dynamic_quantization(model)

# User input prompt
prompt = "The quick brown fox"


# SAMPLE + QUANTISATION OPTIMISATION
inf_times = []
num_runs = 10
with torch.no_grad():
    for i in range(num_runs):
        start_time = time.time()
        output_text = generate_text(model, tokenizer, prompt)
        elapsed_time = time.time()-start_time
        inf_times.append(elapsed_time)
        print(f"Generated Text {i}: {output_text}\n")

print(f'Elapsed time: {sum(inf_times):.3f}s')
ave_time = sum(inf_times)/len(inf_times)
print(f'Ave Elapsed time: {ave_time:.3f}s')

# SAMPLE + QUANTISATION + MULTITHREADING OPTIMISATION
# start_time = time.time()
# # Generate text in parallel
# num_runs = 10
# generated_texts = parallel_text_generation(model, tokenizer, prompt, num_runs=num_runs)
# elapsed_time = time.time() - start_time

# # Display the generated texts
# for i, text in enumerate(generated_texts, 1):
#     print(f"Generated Text {i}: {text}\n")

# print(f'Elapsed time: {elapsed_time:.3f}s')
# ave_time = elapsed_time/num_runs
# print(f'Ave Elapsed time: {ave_time:.3f}s')
