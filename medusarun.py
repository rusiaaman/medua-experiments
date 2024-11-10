import argparse
import os
import re
import sys
import torch
import json
from medusa_model import MedusaModel

model_id = "FasterDecoding/medusa-v1.0-vicuna-7b-v1.5"
model = MedusaModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

tokenizer = model.get_tokenizer()

prompt = "1 + 1 = "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    model.base_model.device
                )
outputs = model.medusa_generate(
        input_ids,
        temperature=1.0,
        max_steps=2,
    )
breakpoint()