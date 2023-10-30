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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

import traceback
import json
import sys


import subprocess
result = subprocess.run(['df', '-kh'], stdout=subprocess.PIPE)
print("====instance df====", str(result.stdout))


tokenizer = AutoTokenizer.from_pretrained("hiyouga/baichuan-7b-sft", trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


def model_fn(model_dir):
    print("====model_fn_Start====")
    
    model = AutoModelForCausalLM.from_pretrained("hiyouga/baichuan-7b-sft",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True)

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
                
        inputs = tokenizer(data, return_tensors='pt').to("cuda")
        # inputs = tokenizer(text, return_tensors='pt').to('cuda')
        pred = model.generate(**inputs,
                              streamer=streamer,
                              **params
                             )
        result = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

        print(f'====result: {result}====')
        
        result_list = result.split("ASSISTANT:")
        if len(result_list) > 1:
            result = result_list[-1].strip() # 如果是多轮对话，则返回最后的，前面的应该是对话历史
        
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



