import numpy as np
import torch
import torch.nn as nn
from medusa_model import MedusaConfig
from modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import PreTrainedModel, PretrainedConfig
import onnxruntime as ort
from utils import generate_candidates, evaluate_posterior, generate_medusa_buffers, update_inference_inputs, reset_medusa_mode
from kv_cache import initialize_past_key_values
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings



def apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share):
    # Check that all model inputs will be provided
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        print(f"The following model inputs are missing: {missing_inputs}")
        # raise Exception("There are missing inputs to the model. Please add them and try again.")

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


def initialize_medusa(input_ids, model: ort.InferenceSession, config, device, use_fp16, medusa_mask):
    bsz = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(
        0, seq_len, dtype=torch.long, device=device
    ).unsqueeze(0).expand(bsz, -1)
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=device)

    
    inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "medusa_mask": torch.ones((1, 1, seq_len, seq_len), dtype=torch.float16, device=device),
    }
    outputs = {
        "logits": torch.zeros((bsz, seq_len, config.vocab_size), dtype=torch.float16, device=device),
        "medusa_logits": torch.zeros((config.medusa_num_heads, bsz, seq_len, config.vocab_size), dtype=torch.float16, device=device),
    }
    io_binding = apply_io_binding(model, inputs, outputs, use_fp16, False)
    io_binding.synchronize_inputs()
    model.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    return outputs["medusa_logits"], outputs["logits"]


def tree_decoding(
    model: ort.InferenceSession,
    config,
    tree_candidates,
    retrieve_indices,
    medusa_attn_mask,
    input_ids,
):
    # tree_medusa_logits, outputs, tree_logits = model(
    #     tree_candidates,
    #     output_orig=True,
    #     position_ids=position_ids,
    #     medusa_forward=True,
    # )
    past_input_ids_len = input_ids.shape[1]
    tree_candidates = torch.cat(
        [input_ids.expand(tree_candidates.shape[0], -1), tree_candidates], dim=1
    )
    bsz = tree_candidates.shape[0]
    seq_len = tree_candidates.shape[1]

    position_ids = torch.arange(
        0, seq_len, dtype=torch.long
    ).unsqueeze(0).expand(bsz, -1)
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.long,)


    # Expand medusa_mask to enable casting
    pad_to = max(0, seq_len - medusa_attn_mask.shape[-1])
    medusa_attn_mask = torch.nn.functional.pad(medusa_attn_mask, (pad_to, 0, pad_to, 0), value=1)
    medusa_attn_mask = medusa_attn_mask[:, :, :seq_len, :seq_len]
    
    inputs = {
        "input_ids": tree_candidates,
        "medusa_mask": medusa_attn_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }

    outputs = {
        "logits": torch.zeros((bsz, seq_len, config.vocab_size), dtype=torch.float16),
        "medusa_logits": torch.zeros((config.medusa_num_heads, bsz, seq_len, config.vocab_size), dtype=torch.float16),
    }
    
    io_binding = apply_io_binding(model, inputs, outputs, True, False)
    io_binding.synchronize_inputs()
    model.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()


    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = outputs['logits'][0, past_input_ids_len + retrieve_indices]
    medusa_logits = outputs['medusa_logits'][:, 0, past_input_ids_len + retrieve_indices]
    return medusa_logits, logits


class MedusaModel:
    

    
    def get_medusa_choice(self, model_name):
        vicuna_7b_stage2 = [(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0), (2, 0), (4,), (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,), (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1), (0, 0, 2, 0), (7, 0), (1, 4)]
        return vicuna_7b_stage2

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device='cpu'
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self.model, self.config, 'cpu', True, medusa_buffers["medusa_attn_mask"]
        )
        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits = tree_decoding(
                self.model, 
                self.config,
                tree_candidates,
                medusa_buffers["retrieve_indices"],
                medusa_buffers["medusa_attn_mask"],
                input_ids
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                logits,
                medusa_logits,
                new_token,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


    def __init__(self, onnx_model_path, sess_options, ep, model_name, cache_dir):
        self.model = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=[ep])

        self.config = MedusaConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.base_model_name_or_path = model_name