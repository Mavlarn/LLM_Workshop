from djl_python import Input, Output
import deepspeed
import torch
import logging
import math
import os

from transformers import LlamaForCausalLM, LlamaTokenizer

model = None
tokenizer = None

from deepspeed import __version__ as vv
logging.info(f"====deepspeed version {vv}")

from transformers import __version__ as v2
logging.info(f"====transformers version {v2}")

def load_model(properties):
    
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties["model_id"]
    logging.info(f"Loading model in {model_location}")
    
    # tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
    tokenizer = LlamaTokenizer.from_pretrained(model_location, legacy=True)


    model = LlamaTokenizer.from_pretrained(
        model_location,
        torch_dtype='fp16',
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)

    logging.info(f"Starting DeepSpeed init with TP={tensor_parallel}")
    model = deepspeed.init_inference(
        model,
        mp_size=tensor_parallel,
        dtype=model.dtype,
        replace_with_kernel_inject=True
    )
    model = model.module
    return model, tokenizer

def run_inference(model, tokenizer, data, history, params):
    print(f"====model chat data: {data}; params: {params}")
    
    inputs = tokenizer(data, return_tensors="pt")  #add_special_tokens=False ?
    generation_output = model.generate(
        input_ids = inputs["input_ids"].to('cuda'),
        attention_mask = inputs['attention_mask'].to('cuda'),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config = generation_config
    )
    response = tokenizer.decode(generation_output, skip_special_tokens=True)
    print(f"====model result: {response}")
    
    result_list = response.split("[/INST]")
    if len(result_list) > 1:
        response = result_list[-1].strip()
    
    return response


def handle(inputs: Input):
    global model, tokenizer
    
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    input_data = inputs.get_as_json()
    if not input_data.get("inputs"):
        return "input field can't be null"

    data = input_data.get("inputs")
    params = input_data.get("parameters",{})
    history = input_data.get('history', [])

    outputs = run_inference(model, tokenizer, data, history, params)
    result = {"outputs": outputs}
    return Output().add_as_json(result)