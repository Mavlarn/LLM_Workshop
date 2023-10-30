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

import requests
import boto3
import sagemaker
import torch


from torch import autocast
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v2")


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text, sample=True, top_p=0.45, temperature=0.7,model=None, history=[]):
    text = preprocess(text)
    print(f"====model chat params: temperature: {temperature}, history: {history}, top_p: {top_p}, text: {text}")
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=1024, return_tensors="pt").to("cuda")
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


def model_fn(model_dir):
    """
    Load the model for inference,load model from os.environ['model_name'],diffult use stabilityai/stable-diffusion-2
    
    """
    print("====model_fn_Start====")
    model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2")
    model = model.half().to("cuda")
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
        if input_data.get('temperature'):
            temperature = input_data['temperature']
        if input_data.get('history'):
            history = input_data['history']
        if input_data.get('top_p'):
            top_p = input_data['top_p']
        
        result=answer(input_data['ask'], top_p=top_p, temperature=temperature, model=model, history=history)
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



