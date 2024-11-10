from transformers import AutoConfig, AutoTokenizer
import numpy as np
import onnxruntime as ort
import torch

model_name = "FasterDecoding/medusa-v1.0-vicuna-7b-v1.5"  # Model name in Hugging Face
onnx_model_path = "/home/arusia/medusa/onnxruntime/onnxruntime/python/tools/transformers/models/medusa/medusa-tiny/rank_0_medusa-v1.0-vicuna-7b-v1.5_decoder_merged_model_fp16.onnx"  # Path to exported ONNX model on disk
use_fp16 = True  # True when KV cache inputs/outputs are in float16
use_buffer_share = True 
cache_dir = '/home/arusia/.cache/huggingface/hub/'

max_iterations = 10  # max(prompt length + generation length)
config = AutoConfig.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir)
torch_dtype = torch.float16 if use_fp16 else torch.float32
prompt = ["""1 + 1 = """]  # Prompt for generation
def get_initial_inputs_and_outputs(config, tokenizer, prompt, device, use_fp16, use_buffer_share):
    tokenizer.pad_token = "[PAD]"  # Set pad token for tokenizer
    encodings_dict = tokenizer.batch_encode_plus(prompt, padding=True)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    # Move inputs from tokenizer to on-device memory
    input_ids = torch.tensor(encodings_dict["input_ids"], device=device, dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], device=device, dtype=torch.int64)
    bsz = input_ids.shape[0]
    sequence_length = input_ids.shape[1]
    inputs = {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "position_ids": torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1).contiguous(),
        "medusa_mask": torch.ones((1, 1, sequence_length, sequence_length), device=device, dtype=torch_dtype).contiguous()
    }
    # Pre-allocate on-device memory for past_key_values (past KV cache)
    # Share on-device memory if use_buffer_share is True
    batch_size, sequence_length = input_ids.shape
    max_sequence_length = config.max_position_embeddings
    num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_attention_heads
    for i in range(config.num_hidden_layers):
        past_key = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        past_value = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        inputs.update({
            f"past_key_values.{i}.key": past_key.contiguous(),
            f"past_key_values.{i}.value": past_value.contiguous()
        })
    
    # Pre-allocate on-device memory for logits
    logits = torch.zeros(batch_size, sequence_length, config.vocab_size, device=device, dtype=torch_dtype)
    outputs = {
        "logits": logits.contiguous()
    }

    # Pre-allocate on-device memory for present KV cache if use_buffer_share is False
    if not use_buffer_share:
        for i in range(config.num_hidden_layers):
            present_key = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            present_value = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            outputs.update({
                f"present.{i}.key": present_key.contiguous(),
                f"present.{i}.value": present_value.contiguous()
            })

    return inputs, outputs

inputs, outputs = get_initial_inputs_and_outputs(config, tokenizer, prompt, 'cpu', use_fp16, use_buffer_share)

sess_options = ort.SessionOptions()
ep = ("CPUExecutionProvider", {})  # change to ep = "CPUExecutionProvider" for CPU
model = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=[ep])

def apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share):
    # Check that all model inputs will be provided
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        print(f"The following model inputs are missing: {missing_inputs}")
        raise Exception("There are missing inputs to the model. Please add them and try again.")

    # Remove unnecessary inputs from model inputs
    unnecessary_inputs = user_inputs - model_inputs
    if len(unnecessary_inputs):
        for unnecessary_input in unnecessary_inputs:
            print(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
            del inputs[unnecessary_input]

    # Bind inputs/outputs to IO binding
    io_binding = model.io_binding()
    device = None
    pt_to_np = {
        "torch.int64": np.int64,
        "torch.float32": np.float32,
        "torch.float16": np.float16
    }

    for k, v in inputs.items():
        io_binding.bind_input(
            name=k,
            device_type=v.device.type,
            device_id=0 if v.device.type == "cpu" else 0,
            element_type=pt_to_np[repr(v.dtype)],
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr()
        )
        device = v.device

    for output in model.get_outputs():
        name = output.name
        if use_buffer_share and "present" in name:
            # Bind KV cache outputs to KV cache inputs
            v = inputs[name.replace("present", "past_key_values")]
            io_binding.bind_output(
                name=name,
                device_type=v.device.type,
                device_id=0,
                element_type=np.float16,
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )
        elif name in outputs:
            v = outputs[name]
            io_binding.bind_output(
                name=name,
                device_type=device.type,
                device_id=0 if device.type == "cpu" else device.index,
                element_type=(np.float16 if use_fp16 else np.float32),
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )

    return io_binding

all_token_ids = inputs["input_ids"].clone()  # store prompt token ids + generated token ids for transcription at the end
batch_size, sequence_length = all_token_ids.shape
max_sequence_length = config.max_position_embeddings
num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_attention_heads

current_length = sequence_length  # keep track of current length (prompt length + generation length)
has_eos = torch.zeros(batch_size, device='cpu', dtype=torch.bool)  # keep track of each batch entry's status and whether it has reached end-of-sequence (EOS) or not

for _ in range(max_iterations):
    # Run inference
    io_binding = apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share)
    io_binding.synchronize_inputs()
    model.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    

    # Sample/choose next token with argmax (greedy search)
    if outputs["logits"].shape[1] > 1:
        prompt_end_indices = inputs["attention_mask"].sum(1) - 1
        idxs = prompt_end_indices.unsqueeze(dim=1).repeat(1, config.vocab_size).view(batch_size, 1, config.vocab_size)
        next_token_logits = torch.gather(outputs["logits"], 1, idxs).squeeze()
    else:
        next_token_logits = outputs["logits"][:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)

    # Check if we previously reached EOS token id or if generated token id is EOS token id
    has_eos = has_eos | next_tokens == tokenizer.eos_token_id

    # Determine which new tokens to add to list of all token ids
    # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
    tokens_to_add = next_tokens.masked_fill(has_eos, tokenizer.eos_token_id).reshape([batch_size, 1])
    all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

    # Return early if:
    # 1) all batch entries have reached EOS token id or 
    # 2) we have reached the max length of a batch entry (prompt length + generation length) or
    # 3) max sequence length that the model can support
    current_length += 1
    if torch.all(has_eos) or current_length > max_sequence_length:
        break

    # Update inputs for next inference run
    inputs["input_ids"] = all_token_ids
    inputs['position_ids'] = torch.arange(all_token_ids.shape[1], device='cpu', dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], (~has_eos).to(torch.int64).reshape(batch_size, 1)], 1)
    inputs['medusa_mask'] = torch.ones((1, 1, all_token_ids.shape[1], all_token_ids.shape[1]), device='cpu', dtype=torch_dtype).contiguous()
    outputs["logits"] = torch.zeros(batch_size, all_token_ids.shape[1], config.vocab_size, device='cpu', dtype=torch_dtype)

    # If buffer sharing is off, pass the present KV cache from previous iteration as the past KV cache for next iteration
    if not use_buffer_share:
        for i in range(config.num_hidden_layers):
            inputs[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            inputs[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

        new_sequence_length = inputs["attention_mask"].shape[1]
        for i in range(config.num_hidden_layers):
            present_key = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device='cpu', dtype=torch_dtype)
            present_value = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device='cpu', dtype=torch_dtype)
            outputs.update({
                f"present.{i}.key": present_key.contiguous(),
                f"present.{i}.value": present_value.contiguous()
            })

    print(tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)
    )