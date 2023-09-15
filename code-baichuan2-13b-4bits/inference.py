import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


import traceback
import json
import sys


import subprocess
result = subprocess.run(['df', '-kh'], stdout=subprocess.PIPE)
print("====instance df====", str(result.stdout))

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat-4bits", use_fast=False, trust_remote_code=True)


def model_fn(model_dir):
    print("====model_fn_Start====")
    model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat-4bits", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat-4bits")

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
        params = input_data.get("parameters", {})

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
    return json.dumps(
        {
            'outputs': prediction
        }
    )



