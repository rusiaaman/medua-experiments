import argparse
import os
import re
import sys
import torch
import json
from medusa_model_onnx import MedusaModel
import onnxruntime as ort

model_id = "FasterDecoding/medusa-v1.0-vicuna-7b-v1.5"
model_onnx_path = "/home/arusia/medusa/onnxruntime/onnxruntime/python/tools/transformers/models/medusa/medusa-tiny-bk/rank_0_medusa-v1.0-vicuna-7b-v1.5_decoder_merged_model_fp16.onnx"  # Path to exported ONNX model on disk
sess_options = ort.SessionOptions()
ep = ("CPUExecutionProvider", {})  # change to ep = "CPUExecutionProvider" for CPU
cache_dir = "../.cache/huggingface/hub"

model = MedusaModel(
            model_onnx_path,
            sess_options=sess_options, ep=ep,
            model_name=model_id,
            cache_dir=cache_dir
        )

tokenizer = model.tokenizer

prompt = "LLM: The capital of India is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    'cpu'
                )
outputs = model.medusa_generate(
        input_ids,
        # temperature=1.0,
        max_steps=4,
    )
for item in outputs:
    print(item['text'])