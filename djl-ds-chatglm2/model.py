from djl_python import Input, Output
import deepspeed
import torch
import logging
import math
import os

from transformers import AutoConfig, AutoTokenizer, AutoModel

model = None
tokenizer = None

from deepspeed import __version__ as vv
logging.info(f"====deepspeed version {vv}")

from transformers import __version__ as v2
logging.info(f"====transformers version {v2}")

def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def load_model(properties):
    
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties["model_id"]
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)

    model = AutoModel.from_pretrained(model_location, trust_remote_code=True).half().cuda()

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
    data = preprocess(data)
    print(f"====model chat data: {data}; params: {params}")
    response, history = model.chat(
        tokenizer,
        data,
        history=history,
        **params
    )
    print(f"====model result: {response}")
    return postprocess(response)


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