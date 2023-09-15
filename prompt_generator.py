from langchain.schema.messages import HumanMessage, AIMessage

def create_payload(input, instruct=None, model_code=None, history=[], parameters=None):
    text = generate_prompt(input, instruct, history, model_code)
    # "parameters": {
    #     "top_p": 0.9,
    #     "temperature": 0.8,
    #     "repetition_penalty": 1.03
    # }
    
    if not parameters:
        # parameters = {}
        parameters = {
            "do_sample": False,
            # "top_p": 0.6,
            "temperature": 0.3,
            "top_k": 50,
            "max_new_tokens": 2048
        }
    payload = {
        "inputs": text,
        "parameters": parameters
    }
    
    if model_code and model_code == 'falcon':
        parameters['stop'] = ["\nUser:","<|endoftext|>","</s>"]
    elif model_code == 'llama2':
        parameters['return_full_text'] = False
        if 'max_new_tokens' not in parameters:
            parameters['max_new_tokens'] = 1024
        parameters['stop'] = ["</s>"]
    
    return payload


def generate_prompt(input, instruct=None, history=[], model_code=None):
    if not model_code:
        return input
    if model_code == 'baichuan':
        return generate_prompt_baichuan(input, instruct)
    if model_code == 'baichuan2':
        return generate_prompt_baichuan2(input, instruct, history)
    elif model_code == 'llama':
        return generate_prompt_llama(input, instruct)
    elif model_code == 'llama2':
        return generate_prompt_llama2(input, instruct)
    elif model_code == 'falcon':
        return generate_prompt_falcon(input, instruct)
    elif model_code == 'cpmbee':
        return generate_prompt_cpmbee(input, instruct)
    return input


def generate_prompt_baichuan(input, instruct):
    instruction = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \nHuman: {}\nAssistant:"""
    text = instruction.format(input)
    return text


def generate_prompt_baichuan2(input, instruct, history=[]):
    messages = []
    if instruct:
        messages.append({"role": 'system', "content": instruct})
    for msg in history:
        if type(msg) == HumanMessage:
            messages.append({"role": 'user', "content": msg.content})
        elif type(msg) == AIMessage:
            messages.append({"role": 'assistant', "content": msg.content})
    messages.append({"role": "user", "content": input})
    return messages


def generate_prompt_llama(input, instruct):
    if instruct:
        input = instruct + '\n' + input
    instruction = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"""
    text = instruction.format(input)
    return text


def generate_prompt_cpmbee(input, instruct):
    text = {
        "question": input,
        "<ans>":""}
    if instruct:
        text['input'] = instruct
    return text


# {"input": "今天天气不错，", "prompt":"问答", "question": "今天天气怎么样", "<ans>":""}
# def generate_prompt_qa_cpmbee(context, question):
#     prompt_template = {
#         "input": context,
#         "prompt": "问答",
#         "question": question,
#         "<ans>":""
#     }
#     return prompt_template


def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt


def generate_prompt_llama2(input, instruct):
    if instruct:
        instruct = "You are a helpful assistant. 你是一个乐于助人的助手。\n" + instruct
    else:
        instruct = "You are a helpful assistant. 你是一个乐于助人的助手。"
    messages = [
        {
            "role": "system",
            "content": instruct
        },
        {
            "role": "user",
            "content": input
        }
    ]
    text = build_llama2_prompt(messages)
    return text


def generate_prompt_falcon(input, instruct):
    text = f"You are an helpful Assistant, called Falcon. Answer user's question as best as you can. And answer in Chinese.\n\nUser: {input}\nFalcon:"
    return text


def generate_prompt_qa(model_code=None):
    if model_code == 'baichuan':
        return generate_prompt_qa_baichuan(input)
    elif model_code == 'cpmbee':
        return generate_prompt_qa_cpmbee(input)
    else:
        prompt_template = "使用下面的已知内容，简洁、准确的回答最后的问题，并使用中文。如果你无法从已知内容获取答案，就直接返回'根据已知信息无法回答该问题。'，不要编造答案。\n\n已知内容:\n{context}\n\n问题: {question}\n答案:"""
        return prompt_template

def generate_prompt_qa_baichuan(context, question):
    prompt_template = "使用下面的已知内容，简洁、准确的回答最后的问题，并使用中文。如果你无法从已知内容获取答案，就直接返回'根据已知信息无法回答该问题。'，不要编造答案。\n\n已知内容:\n{context}\n\nUSER: {question}\nASSISTANT:"""
    return prompt_template



