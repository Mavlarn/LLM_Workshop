from djl_python import Input, Output
import torch
import logging
import math
import os, json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
STOP_flag = "[DONE]"

model = None
tokenizer = None
generator = None

def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    tokenizer = AutoTokenizer.from_pretrained(model_location, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_location, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)    
    model.generation_config = GenerationConfig.from_pretrained(model_location)
    return model, tokenizer


def construct_message(history, prompt):
    message = []
    for question, answer in history:
        message.append({"role":"user","content":question})
        message.append({"role":"assistant","content":answer})
    message.append({"role":"user","content":prompt})
    return message

def stream_items(messages):
    global model, tokenizer
    size = 0
    response = ""
    
    res_generator = model.chat(tokenizer, messages,stream=True)
    for response in res_generator:
        this_response = response[size:]
        size = len(response)
        stream_buffer = { "outputs":this_response,"finished": False}
        yield stream_buffer
    ## stop
    # yield {"query": prompt, "outputs": STOP_flag, "response": response, "history": [], "finished": True}
    


def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    print(f'input data: {data}')
    
    input_sentences = data["inputs"]
    if type(input_sentences) == str:
        input_sentences = json.loads(input_sentences)
    params = data.get("parameters", {})
    # print(f'inputs: {input_sentences}, type is: {type(input_sentences)}')
    
    if params.get("max_new_tokens"):
        model.generation_config.max_new_tokens = params.get("max_new_tokens")

    stream = data.get('stream', False)  
    outputs = Output()
    if stream:
        outputs.add_property("content-type", "application/jsonlines")
        outputs.add_stream_content(stream_items(input_sentences))
    else:
        response = model.chat(tokenizer, input_sentences, stream=False)
        result = {"outputs": response}
        outputs.add_as_json(result)
        
    return outputs
