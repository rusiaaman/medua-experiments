from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import LlamaConfig, LlamaTokenizer
import numpy as np
import onnxruntime as ort
import torch
from contextlib import asynccontextmanager

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 64

class GenerationResponse(BaseModel):
    generated_texts: str

model_name = "meta-llama/Llama-2-7b-hf"
onnx_model_path = "./llama2-7b-fp16-gqa/rank_0_vicuna-7b-v1.5_decoder_merged_model_fp16.onnx"
cache_dir = './cache_dir'
config = LlamaConfig.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir)
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True, cache_dir=cache_dir)

sess_options = ort.SessionOptions()
ep = ("CPUExecutionProvider", {})
model = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=[ep])

app = FastAPI()

def get_initial_inputs_and_outputs(config, tokenizer, prompt, device, use_fp16, use_buffer_share):
    tokenizer.pad_token = "[PAD]"
    encodings_dict = tokenizer.batch_encode_plus(prompt, padding=True)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    input_ids = torch.tensor(encodings_dict["input_ids"], device=device, dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], device=device, dtype=torch.int64)
    inputs = {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
    }

    batch_size, sequence_length = input_ids.shape
    max_sequence_length = config.max_position_embeddings
    num_heads = config.num_key_value_heads
    head_size = config.hidden_size // config.num_attention_heads
    
    for i in range(config.num_hidden_layers):
        past_key = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, 
                             device=device, dtype=torch_dtype)
        past_value = torch.zeros(batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, 
                               device=device, dtype=torch_dtype)
        inputs.update({
            f"past_key_values.{i}.key": past_key.contiguous(),
            f"past_key_values.{i}.value": past_value.contiguous()
        })

    logits = torch.zeros(batch_size, sequence_length, config.vocab_size, device=device, dtype=torch_dtype)
    outputs = {"logits": logits.contiguous()}

    if not use_buffer_share:
        for i in range(config.num_hidden_layers):
            present_key = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            present_value = torch.zeros(batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            outputs.update({
                f"present.{i}.key": present_key.contiguous(),
                f"present.{i}.value": present_value.contiguous()
            })

    return inputs, outputs

def apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share):
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        raise ValueError(f"Missing model inputs: {missing_inputs}")

    unnecessary_inputs = user_inputs - model_inputs
    for unnecessary_input in unnecessary_inputs:
        del inputs[unnecessary_input]

    io_binding = model.io_binding()
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

    for output in model.get_outputs():
        name = output.name
        if use_buffer_share and "present" in name:
            v = inputs[name.replace("present", "past_key_values")]
            io_binding.bind_output(
                name=name,
                device_type=v.device.type,
                device_id=0 if v.device.type == "cpu" else 0,
                element_type=np.float16,
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )
        else:
            v = outputs[name]
            io_binding.bind_output(
                name=name,
                device_type=v.device.type,
                device_id=0 if v.device.type == "cpu" else 0,
                element_type=(np.float16 if use_fp16 else np.float32),
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )

    return io_binding

async def generate_text(prompts: List[str], max_length: int = 64) -> List[str]:
    use_fp16 = True
    use_buffer_share = True
    device = 'cpu'
    
    inputs, outputs = get_initial_inputs_and_outputs(config, tokenizer, , device, use_fp16, use_buffer_share)
    
    all_token_ids = inputs["input_ids"].clone()
    batch_size, sequence_length = all_token_ids.shape
    max_sequence_length = config.max_position_embeddings
    num_heads = config.num_key_value_heads
    head_size = config.hidden_size // config.num_attention_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    current_length = sequence_length
    has_eos = torch.zeros(batch_size, device=device, dtype=torch.bool)

    while current_length <= max_length:
        io_binding = apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share)
        io_binding.synchronize_inputs()
        model.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        if outputs["logits"].shape[1] > 1:
            prompt_end_indices = inputs["attention_mask"].sum(1) - 1
            idxs = prompt_end_indices.unsqueeze(dim=1).repeat(1, config.vocab_size).view(batch_size, 1, config.vocab_size)
            next_token_logits = torch.gather(outputs["logits"], 1, idxs).squeeze()
        else:
            next_token_logits = outputs["logits"][:, -1, :]
        
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        has_eos = has_eos | (next_tokens == tokenizer.eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, tokenizer.eos_token_id).reshape([batch_size, 1])
        all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

        current_length += 1
        if torch.all(has_eos) or current_length > max_length or current_length > max_sequence_length:
            break

        inputs["input_ids"] = tokens_to_add
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], (~has_eos).to(torch.int64).reshape(batch_size, 1)], 1)

        if outputs["logits"].shape[1] != 1:
            outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
        outputs["logits"].zero_()

        if not use_buffer_share:
            for i in range(config.num_hidden_layers):
                inputs[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                inputs[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

            new_sequence_length = inputs["attention_mask"].shape[1]
            for i in range(config.num_hidden_layers):
                present_key = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device=device, dtype=torch_dtype)
                present_value = torch.zeros(batch_size, num_heads, new_sequence_length, head_size, device=device, dtype=torch_dtype)
                outputs.update({
                    f"present.{i}.key": present_key.contiguous(),
                    f"present.{i}.value": present_value.contiguous()
                })

    return tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        generated_texts = await generate_text(request.prompts, request.max_length)
        return GenerationResponse(generated_texts=generated_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if tokenizer is None or model is None or config is None:
        raise HTTPException(status_code=503, detail="Model components not loaded")
    return {"status": "healthy"}


