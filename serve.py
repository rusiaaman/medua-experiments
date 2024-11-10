from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import onnxruntime as ort
import torch
from transformers import LlamaConfig, LlamaTokenizer
from contextlib import asynccontextmanager

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int

class GenerationResponse(BaseModel):
    generated_text: str

@dataclass
class QueueItem:
    prompt: str
    max_length: int
    future: asyncio.Future

class BatchProcessor:
    def __init__(self, batch_size=4, max_wait_time=0.5):
        self.queue = asyncio.Queue[QueueItem]()
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.is_running = False
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.onnx_model_path = "./llama2-7b-fp16-gqa/rank_0_vicuna-7b-v1.5_decoder_merged_model_fp16.onnx"
        self.cache_dir = './cache_dir'
        
        # Initialize model components
        self.config = LlamaConfig.from_pretrained(
            self.model_name, 
            use_auth_token=True, 
            cache_dir=self.cache_dir
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.model_name, 
            use_auth_token=True, 
            cache_dir=self.cache_dir
        )
        
        sess_options = ort.SessionOptions()
        ep = ("CPUExecutionProvider", {})
        self.model = ort.InferenceSession(
            self.onnx_model_path, 
            sess_options=sess_options, 
            providers=[ep]
        )

    async def start(self):
        """Start the batch processor worker"""
        self.is_running = True
        asyncio.create_task(self._process_batches())

    async def stop(self):
        """Stop the batch processor worker"""
        self.is_running = False
        
    async def add_to_queue(self, prompt: str, max_length: int) -> asyncio.Future[str]:
        """Add a request to the queue and return a future for the result"""
        future = asyncio.Future[str]()
        request_id = id(future)
        item = QueueItem(
            prompt=prompt,
            max_length=max_length,
            future=future,
        )
        await self.queue.put(item)
        return future

    async def _process_batches(self):
        """Main worker loop that processes batches of requests"""
        while self.is_running:
            batch: list[QueueItem] = []
            try:
                # Get the first item and start the batch
                first_item = await self.queue.get()
                batch.append(first_item)
                batch_deadline = datetime.now() + timedelta(seconds=self.max_wait_time)

                # Try to fill the batch
                while len(batch) < self.batch_size and datetime.now() < batch_deadline:
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=(batch_deadline - datetime.now()).total_seconds()
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                print(len(batch))
                # Process the batch
                prompts = [item.prompt for item in batch]
                max_length = max(item.max_length for item in batch)
                
                try:
                    results = await generate_text(
                        prompts=prompts,
                        max_length=max_length,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=self.config
                    )
                    
                    # Set results for each future
                    for item, result in zip(batch, results):
                        if not item.future.done():
                            item.future.set_result(result)
                            
                except Exception as e:
                    # Handle errors by setting exception for all futures in batch
                    for item in batch:
                        if not item.future.done():
                            item.future.set_exception(e)
                            
            except Exception as e:
                # Handle unexpected errors in the worker loop
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
            finally:
                # Mark all items in batch as done
                for _ in batch:
                    self.queue.task_done()

# Initialize FastAPI app and batch processor
batch_processor = BatchProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await batch_processor.start()
    yield
    await batch_processor.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        future = await batch_processor.add_to_queue(
            prompt=request.prompt,
            max_length=request.max_length
        )
        result = await future
        return GenerationResponse(generated_text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

# Modify the generate_text function to accept model components as parameters
async def generate_text(
    prompts: List[str],
    max_length: int,
    model: ort.InferenceSession,
    tokenizer: LlamaTokenizer,
    config: LlamaConfig
) -> List[str]:
    use_fp16 = True
    use_buffer_share = True
    device = 'cpu'
    
    inputs, outputs = get_initial_inputs_and_outputs(
        config, tokenizer, prompts, device, use_fp16, use_buffer_share
    )
    
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


    