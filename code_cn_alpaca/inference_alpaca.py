# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

import traceback
import json
import sys

tokenizer = LlamaTokenizer.from_pretrained('shibing624/chinese-alpaca-plus-13b-hf')

import subprocess
result = subprocess.run(['df', '-kh'], stdout=subprocess.PIPE)
print("====instance df====", str(result.stdout))

    
def answer(text, sample=True, top_p=0.45, temperature=0.7,model=None, top_k=40, repetition_penalty=1.15):
    print(f"====model chat params: temperature: {temperature}, top_k: {top_k}, top_p: {top_p}, text: {text}")
    input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda')
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=128,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def model_fn(model_dir):
    print("====model_fn_Start====")
    model = LlamaForCausalLM.from_pretrained(
        'shibing624/chinese-alpaca-plus-13b-hf',
        cache_dir="/tmp/model_cache/",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    model.eval()
    print("====model_fn_End====")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    # {
    # "ask": "写一个文章，题目是未来城市"
    # }
    input_data = json.loads(request_body)
    if 'ask' not in input_data:
        input_data['ask']="写一个文章，题目是未来城市"
    return input_data


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """   
    print('predict_fn input_data: ', input_data)

    try:        
        params = {}
        # text, sample=True, top_p=0.45, temperature=0.7,model=None, history=[]):
        temperature = 0.7
        history = []
        top_p = 0.45
        top_k = 40
        repetition_penalty=1.15
        if input_data.get('temperature'):
            temperature = input_data['temperature']
        if input_data.get('top_k'):
            top_k = input_data['top_k']
        if input_data.get('top_p'):
            top_p = input_data['top_p']
        if input_data.get('repetition_penalty'):
            repetition_penalty = input_data['repetition_penalty']
        
        result=answer(input_data['ask'], top_p=top_p, temperature=temperature, model=model, top_k=top_k, repetition_penalty=repetition_penalty)
        print(f'====result: {result}====')
        
        result_list = result.split("### Response:")
        if len(result_list) > 1:
            return result_list[1].strip()
        else:
            return result
        
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"====Exception===={ex}")

    return 'Not found answer'


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    return json.dumps(
        {
            'answer': prediction
        }
    )



