import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import torch
import onnxruntime as ort
from medusa_model_onnx import MedusaModel
from contextlib import asynccontextmanager

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_steps: int = 2
    temperature: Optional[float] = 1.0

class GenerationResponse(BaseModel):
    generated_text: str

@dataclass
class QueueItem:
    prompt: str
    max_steps: int
    temperature: float
    future: asyncio.Future

class BatchProcessor:
    def __init__(self, batch_size=1, max_wait_time=0.5):
        self.queue = asyncio.Queue[QueueItem]()
        self.batch_size = batch_size
        if batch_size > 1:
            raise NotImplementedError("Batch size > 1 is not supported in medusa")
        self.max_wait_time = max_wait_time
        self.is_running = False
        
        # Model initialization
        self.model_id = "FasterDecoding/medusa-v1.0-vicuna-7b-v1.5"
        self.model_onnx_path ="onnx-medusa/medusa-onnx/rank_0_medusa-v1.0-vicuna-7b-v1.5_decoder_merged_model_fp16.onnx"
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        
        # Initialize model components
        sess_options = ort.SessionOptions()
        ep = ("CPUExecutionProvider", {})
        
        self.model = MedusaModel(
            self.model_onnx_path,
            sess_options=sess_options,
            ep=ep,
            model_name=self.model_id,
            cache_dir=self.cache_dir
        )
        
        self.tokenizer = self.model.tokenizer

    async def start(self):
        """Start the batch processor worker"""
        self.is_running = True
        asyncio.create_task(self._process_batches())

    async def stop(self):
        """Stop the batch processor worker"""
        self.is_running = False
        
    async def add_to_queue(self, prompt: str, max_steps: int, temperature: float) -> asyncio.Future[List[str]]:
        """Add a request to the queue and return a future for the result"""
        future = asyncio.Future[List[str]]()
        item = QueueItem(
            prompt=prompt,
            max_steps=max_steps,
            temperature=temperature,
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

                # Process the batch
                try:
                    results = await self._generate_batch(batch)
                    
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

    async def _generate_batch(self, batch: List[QueueItem]) -> List[List[str]]:
        """Process a batch of generation requests"""
        # Convert to async to prevent blocking
        def _generate():
            results = []
            for item in batch:
                input_ids = self.tokenizer.encode(
                    item.prompt,
                    return_tensors="pt"
                ).to('cpu')
                
                outputs = self.model.medusa_generate(
                    input_ids,
                    temperature=item.temperature,
                    max_steps=item.max_steps,
                )
                
                # Extract texts from outputs
                text = [output['text'] for output in outputs][-1]
                results.append(text)
            return results

        # Run in thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(None, _generate)

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
            max_steps=request.max_steps,
            temperature=request.temperature
        )
        result = await future
        return GenerationResponse(generated_text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))