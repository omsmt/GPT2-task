import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import torch
import numpy as np

# Initialize the model and tokenizer
model_name = "gpt2-medium"
set_seed(15) # For reproducibility
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
pad_token_id = tokenizer.eos_token_id
input_text = "Golden Retrievers are a breed of "
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

def warmup(model, input_ids, attention_mask, pad_token_id):
    start_time = time.time()
    outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id)
    elapsed_time = time.time() - start_time
    print(f'Warm up time: {elapsed_time:.3f}s')

def generate_text_and_measure_time(model, input_ids, attention_mask, pad_token_id, **generate_kwargs):
    start_time = time.time()
    outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, **generate_kwargs)
    elapsed_time = time.time() - start_time
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return elapsed_time, generated_text

def perform_experiments(name, num_trials=10, verbose=True, **generate_kwargs):
    times = []
    for _ in range(num_trials):
        elapsed_time, generated_text = generate_text_and_measure_time(model, input_ids, attention_mask, pad_token_id, **generate_kwargs)
        times.append(elapsed_time)
        if verbose:
            print(f"{name} - Generated text: {generated_text}")
    average_time = np.mean(times)
    print(f"{name} - Average Inference Time: {average_time:.3f}s")
    return average_time

# Define different configurations for each decoding strategy
decoding_strategies = {
    "Greedy Search": {"max_new_tokens": 25},
    "Contrastive Search": {"max_new_tokens": 25, "penalty_alpha": 0.6, "top_k": 4 },
    "Sampling": {"max_new_tokens": 25, "do_sample": True, "top_p": 0.95, "top_k": 60},
    "Diverse Beam Search": {"max_new_tokens": 25, "num_beams": 5, "num_beam_groups": 5,  "diversity_penalty": 1.0},
    "Top P Nucleus Sampling": {"max_new_tokens": 25, "do_sample": True, "top_p": 0.92, "top_k": 0},
}

# Model warm up
warmup(model, input_ids, attention_mask, pad_token_id)

# Perform experiments and store average times
average_times = {name: perform_experiments(name, **kwargs) for name, kwargs in decoding_strategies.items()}

# Print averages of all methods
print("\nAverage inference times for all methods:")
for method, avg_time in average_times.items():
    print(f"{method}: {avg_time:.3f}s")

# Summary and Recommendation
best_method = min(average_times, key=average_times.get)
print(f"\nBest method for quick decoding: {best_method} (Average Time: {average_times[best_method]:.3f}s)")


# Results for my computer:

# GPT2
"""Average inference times for all methods:
Greedy Search: 0.790s
Contrastive Search: 1.187s
Sampling: 0.807s
Diverse Beam Search: 1.260s
Top P Nucleus Sampling: 0.907s"""

# GPT2-MEDIUM
"""Average inference times for all methods:
Greedy Search: 2.016s
Contrastive Search: 3.202s
Sampling: 2.062s
Diverse Beam Search: 3.210s
Top P Nucleus Sampling: 2.196s"""

#GPT2-LARGE
"""Average inference times for all methods:
Greedy Search: 3.948s
Contrastive Search: 6.355s
Sampling: 4.093s
Diverse Beam Search: 6.355s
Top P Nucleus Sampling: 4.348s
"""

# Discussion
""" Qualitative result shows that the sampling methods tend to do better in terms of accuracy. Sentences are more coherent, and less likelihood of repeated text.
In terms of model size used, GPT2-Medium is a good mid-ground between accuracy and speed.
As expected, the larger the model, the longer the inference but it's not necessarily bounds better for accuracy. So in this case, it is appropriate to make the trade-off from small to medium
model for an accuracy boost with not too much of a time sacrifice. 
"""