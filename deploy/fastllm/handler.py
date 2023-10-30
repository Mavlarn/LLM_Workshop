import json
import os
from os import environ
import time
from fastllm_pytools import llm


# 这是原来的程序，通过huggingface接口创建模型
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code = True)
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code = True)

# 加入下面这两行，将huggingface模型转换成fastllm模型
# 目前from_hf接口只能接受原始模型，或者ChatGLM的int4, int8量化模型，暂时不能转换其它量化模型
# from fastllm_pytools import llm
# model = llm.from_hf(model, tokenizer, dtype = "float16") # dtype支持 "float16", "int8", "int4"


if "MODEL_PATH" not in environ:
    environ["MODEL_PATH"] = "/opt/ml/model"

model_path = environ['MODEL_PATH']

# if "MODEL_S3_URL" not in environ:
#     environ["MODEL_S3_URL"] = "s3://sagemaker-us-east-1-568765279027/mt_models_uploaded/fastllm--chatglm2-6b/chatglm2_fastllm_model.flm"
    
# model_s3_url = environ['MODEL_S3_URL']
# model_path = os.path.join(model_path, 'model.flm')

print('Getting FastLLM Model from S3')
# print("To: ", model_path)

os.system("chmod +x ./s5cmd")
os.system("./s5cmd cp {0} {1}".format('s3://sagemaker-us-east-1-568765279027/mt_models_uploaded/fastllm-chatglm-6b-fp16/chatglm-6b-fp16.flm', 'model.flm'))

print('Loading Model...')
model = llm.model("model.flm"); # 导入fastllm模型
print('Load FastLLM Model Done...')


def handler(event, context):
    print('event: ', event)
    if type(event) == str:
        input_data = json.loads(event)
    else:
        input_data = event
    if not input_data.get("inputs"):
        return "input field can't be null"

    data = input_data.get("inputs")
    params = input_data.get("parameters",{})
    history = input_data.get('history', [])

    result = model.response(data)
    print('result: ', result)
    
    response = json.dumps({'outputs': result})
    
    return {
        'statusCode': 200,
        'body': response
    }