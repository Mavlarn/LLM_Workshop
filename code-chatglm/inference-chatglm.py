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
import uuid
import io
import sys

import traceback

from PIL import Image

import requests
import boto3
import sagemaker
import torch


from torch import autocast
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text, sample=True, top_k=6, top_p=0.45, temperature=0.7,model=None, history=[]):
    text = preprocess(text)
    print(f"====model chat params: temperature: {temperature}, history: {history}, top_p: {top_p}, text: {text}, top_k: {top_k}")
    response, history = model.chat(
        tokenizer,
        text,
        history=history,
        max_length=6000,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )
    return postprocess(response)


def model_fn(model_dir):
    """
    Load the model for inference,load model from os.environ['model_name'],diffult use stabilityai/stable-diffusion-2
    
    """
    print("====model_fn_Start====")
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    #model = model.to("cuda")
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
        top_k = 6
        if input_data.get('temperature'):
            temperature = input_data['temperature']
        if input_data.get('history'):
            history = input_data['history']
        if input_data.get('top_p'):
            top_p = input_data['top_p']
        if input_data.get('top_k'):
            top_k = input_data['top_k']
        
        result=answer(input_data['ask'], top_p=top_p, top_k=top_k, temperature=temperature, model=model, history=history)
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
            'answer': prediction
        }
    )



