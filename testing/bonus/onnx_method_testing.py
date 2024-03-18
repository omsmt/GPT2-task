from onnxruntime.transformers.models.gpt2.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
from onnxruntime.transformers.io_binding_helper import IOBindingHelper
import onnxruntime
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import numpy
import time

# Terminal command ran to generate the onnx file 
# python -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m "gpt2-medium" --output "testing/bonus/gpt2-medium.onnx" -o -p fp32 -t 10 >export_output.txt 2>&1

# Model and tokenizer setup
model_name = "gpt2-medium"
onnx_model_path = "testing/bonus/gpt2-medium.onnx"
device = torch.device("cpu")

config = GPT2Config.from_pretrained(model_name)
model = MyGPT2LMHeadModel.from_pretrained(model_name)
torch_model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
model.eval().to(device)
torch_model.eval().to(device)
num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd
num_layer = model.config.n_layer

# Utility Functions
def get_tokenizer(path):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_example_inputs(prompt_text):
    tokenizer = get_tokenizer(model_name)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    empty_past = []
    batch_size = input_ids.size(0)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for _ in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past

# Inference Functions
def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(
        batch_size=input_ids.size(0),
        past_sequence_length=past[0].size(3),
        sequence_length=input_ids.size(1),
        config=config,
    )
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device)

    io_binding = IOBindingHelper.prepare_io_binding(
        session, input_ids, position_ids, attention_mask, past, output_buffers, output_shapes
    )
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes, return_numpy=False)
    return outputs

def test_generation(tokenizer, input_text, ort_session=None, num_tokens_to_produce=30):
    model_type = "OnnxRuntime" if ort_session is not None else "PyTorch"
    print(f"\n{'=' * 20}\nText Generation Test\nModel: {model_type}\n{'=' * 20}")
    assert len(input_text) == 1  # This function requires batch_size==1
    use_onnxruntime = ort_session is not None
    eos_token_id = tokenizer.eos_token_id

    input_ids, attention_mask, position_ids, past = get_example_inputs(input_text)
    batch_size = input_ids.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    start_time = time.time()
    for _ in range(num_tokens_to_produce):
        if ort_session is not None:
            outputs = inference_with_io_binding(ort_session, config, input_ids, position_ids, attention_mask, past)

        else:
            outputs = torch_model(
                input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past
            )

        next_token_logits = outputs[0][:, -1, :]
        # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        # Update input_ids, attention_mask, position_ids and past
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)
        position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)

        past = []
        if not use_onnxruntime:
            past = list(outputs[1])  # past in torch output is tuple
        else:
            for i in range(num_layer):
                past_i = (
                    torch.from_numpy(outputs[i + 1])
                    if isinstance(outputs[i + 1], numpy.ndarray)
                    else outputs[i + 1].clone().detach()
                )
                past.append(past_i.to(device))

        if torch.all(has_eos):
            break
    elapsed_time = time.time() - start_time
    print(f'elapsed time: {elapsed_time:.3f}s')

    print("\nGenerated Text:\n" + "-" * 20)
    for i, output in enumerate(all_token_ids):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"{generated_text}\n{'-' * 20}\n")


def max_logits_difference(prompt):
    input_ids, attention_mask, position_ids, empty_past = get_example_inputs(prompt)

    with torch.no_grad():
        torch_output = torch_model(
            input_ids, past_key_values=empty_past, attention_mask=attention_mask, position_ids=position_ids
        )

    session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": numpy.ascontiguousarray(input_ids.cpu().numpy()),
        "attention_mask": numpy.ascontiguousarray(attention_mask.cpu().numpy()),
        "position_ids": numpy.ascontiguousarray(position_ids.cpu().numpy()),
    }
    for i, past_i in enumerate(empty_past):
        ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())
    ort_outputs = session.run(None, ort_inputs)

    logits_masked_diff = (torch_output[0] - ort_outputs[0]) * attention_mask.unsqueeze(2)
    max_logits_diff = logits_masked_diff.abs().max()
    print(f"\nMax Logits Difference (Ignored Padding): {max_logits_diff:.5f}\n")


if __name__ == "__main__":

    tokenizer = get_tokenizer(model_name)
    prompt = ["The quick brown fox jumps"]
    
    # Max logits difference
    max_logits_difference(prompt)    
    
    # Test text generation with ONNX Runtime model
    session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    start_fn_time = time.time()
    test_generation(tokenizer, prompt, ort_session=session)
    elapsed_time_onnx = time.time() - start_fn_time
    
    # Test text generation with PyTorch model
    start_fn_time2 = time.time()
    test_generation(tokenizer, prompt)
    elapsed_time_pytorch = time.time() - start_fn_time2
    
    # Execution Time Summary
    print(f"\n{'=' * 20}\nExecution Time Summary\n{'=' * 20}")
    print(f"ONNX Model Time: {elapsed_time_onnx:.3f}s")
    print(f"PyTorch Model Time: {elapsed_time_pytorch:.3f}s\n")
