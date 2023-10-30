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

import os
import json
import io
import sys

import traceback
import torch

from transformers import AutoTokenizer, AutoModel, AutoConfig

model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "THUDM/chatglm2-6b")
pre_seq_len = int(os.getenv("PRE_SEQ_LEN", 128))
finetune_model_name_or_path = os.getenv("FINETUNE_MODEL_NAME_OR_PATH", "checkpoint-50/pytorch_model.bin")

if "s3" in model_name_or_path:
    os.system("cp ./code/s5cmd  /tmp/ && chmod +x /tmp/s5cmd")
    os.system("/tmp/s5cmd sync {0} {1}".format(model_name_or_path + "*", "/tmp/orignal/"))
    model_name_or_path = "/tmp/orignal/"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text, sample=True, top_k=6, top_p=0.45, temperature=0.7,model=None, max_length=8192, history=[]):
    text = preprocess(text)
    print(f"====model chat params: temperature: {temperature}, history: {history}, top_p: {top_p}, text: {text}, top_k: {top_k}")
    response, history = model.chat(
        tokenizer,
        text,
        history=history,
        max_length=8192,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )
    return postprocess(response)


def model_fn(model_dir):
    """
    Load the model for inference
    
    """
    print("====model_fn_Start====")
    pre_seq_len = int(os.getenv("PRE_SEQ_LEN", 128))
    finetune_model_name_or_path = os.getenv("FINETUNE_MODEL_NAME_OR_PATH", "checkpoint-50/pytorch_model.bin")

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, pre_seq_len=pre_seq_len)
    model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)

    os.system("/tmp/s5cmd sync {0} {1}".format(finetune_model_name_or_path, "/tmp/chatglm2-finetune/"))

    prefix_state_dict = torch.load("/tmp/chatglm2-finetune/pytorch_model.bin")
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    model.half().cuda()

    print("====model_fn_End====")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    # {
    # "ask": "写一个文章，题目是未来城市"
    # }
    print(f"====input_fn===={request_content_type} - {request_body}")
    input_data = json.loads(request_body)
    return input_data


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """   
    try:        
        params = {}
        if 'inputs' not in input_data:
            return "Please provide `inputs`."
        if input_data.get('parameters'):
            params = input_data['parameters']
        history = []
        if input_data.get('history'):
            history = input_data['history']

        result = answer(input_data['inputs'], model=model, history=history, **params)
        print(f'====result: {result}====')
        return result
        
    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"====Exception===={ex}")

    return 'Not found answer'


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    print(content_type)
    return json.dumps(
        {
            'outputs': prediction
        }
    )



