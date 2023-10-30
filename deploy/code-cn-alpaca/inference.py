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

tokenizer = LlamaTokenizer.from_pretrained('shibing624/chinese-alpaca-plus-7b-hf')

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
        'shibing624/chinese-alpaca-plus-7b-hf',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    model.eval()
    # model = LlamaForCausalLM.from_pretrained('shibing624/chinese-alpaca-plus-7b-hf').half().cuda()
    # model.eval()
    print("====model_fn_End====")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    input_data = json.loads(request_body)
    return input_data


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """   
    print('predict_fn input_data: ', input_data)

    try:
        if not input_data.get("inputs"):
            return "input field can't be null"

        data = input_data.get("inputs")
        params = input_data.get("parameters",{})
        # top_p=0.45, top_k = 40, temperature=0.7,repetition_penalty=1.15
        
        result=answer(data, model=model, **params)
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
            'outputs': prediction
        }
    )



