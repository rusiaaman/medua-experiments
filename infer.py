import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

def load_model_and_tokenizer(model_path):
    """
    Load the ONNX model and associated tokenizer
    """
    try:
        from pathlib import Path
        model_path = Path(model_path)
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
        # Load the tokenizer from the same path
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not (model_path / "config.json").exists():
            config = AutoConfig.from_pretrained(model_id)
            # config.save_pretrained(model_path)
        else:
            config = AutoConfig.from_pretrained(model_path)

        # if not (model_path / "generation_config.json").exists():
        #     gen_config = GenerationConfig.from_pretrained(model_id, do_sample=True)
        #     gen_config.save_pretrained(model_path)
        # else:
        #     gen_config = GenerationConfig.from_pretrained(model_path)
        # Load the model using ONNX Runtime
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
        
        # breakpoint()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model and tokenizer: {str(e)}")
        raise e

def generate_completion(prompt, model, tokenizer, max_length=100, temperature=0.7):
    """
    Generate completion for the given prompt
    """
    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate the completion
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,
        )
        
        # Decode the generated tokens
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion
    
    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='CLI tool for text generation using ONNX Runtime')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the local ONNX model directory')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100,
                      help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for text generation')
    
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # Generate completion
        print("\nGenerating completion...\n")
        completion = generate_completion(
            args.prompt,
            model,
            tokenizer,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # Print result
        print("Input prompt:", args.prompt)
        print("\nCompletion:", completion)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    
    return 0

if __name__ == "__main__":
    exit(main())
