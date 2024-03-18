import torch
from torch.quantization import quantize_dynamic
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
set_seed(42)

# torch.set_num_threads(1)

class GPT2_model():
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.tokenizer = None

    def load_model_and_weights(self, model_size, quantise=True):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_size)
            self.model = GPT2LMHeadModel.from_pretrained(model_size).eval()
            if quantise:
                self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.model.to(self.device)

            # Warm up
            input_ids = self.tokenizer.encode("The quick brown fox", return_tensors="pt")
            pad_token_id = self.tokenizer.eos_token_id
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)
            with torch.no_grad():
                self.model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_new_tokens=25, do_sample=True, top_p=0.95, top_k=60)

        except Exception as e:
            print(f"Failed to load model: {e}")

    def inference(self, prompt="The quick brown fox"):
        # Ensure the prompt is not empty
        if not prompt.strip():
            return "Prompt is empty. Please provide a valid input."
        
        # Encode the prompt into tokens
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Assuming no padding is needed initially

        # Set pad_token_id to eos_token_id (50256 for GPT-2)
        pad_token_id = self.tokenizer.eos_token_id

        start_time = time.time()
        # Generate text using autoregressive decoding
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                max_new_tokens=25,
                do_sample=True,  # Enables sampling-based decoding
                top_p=0.95,      # Use nucleus sampling
                top_k=60         # Limits the number of highest probability vocabulary tokens considered for each step
            )
        end_time = time.time()

        # Decode generated tokens to string
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        inf_time = end_time-start_time
        
        return generated_text, inf_time