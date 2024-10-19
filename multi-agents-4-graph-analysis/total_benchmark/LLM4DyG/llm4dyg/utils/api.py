import openai
import os

def send_to_chat(
    model: str, messages: list, max_tokens: int = 2048, temperature: float = 0.5
) -> str:
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:8000/v1"

    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    response_content = response.choices[0].message.content
    d = response['usage'].to_dict()
    d['content'] = response_content
    return d

def send_to_chat_gpt(
    model: str, messages: str, max_tokens: int = 2048, temperature: float = 0.5
) -> str:
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    response_content = response.choices[0].message.content
    d = response['usage'].to_dict()
    d['content'] = response_content
    return d

def send_to_chat_gpt_intruct(
    model: str, messages: str, max_tokens: int = 2048, temperature: float = 0.5
) -> str:
    openai.api_key = os.environ['OPENAI_API_KEY']
    # response = openai.ChatCompletion.create(
    #     model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    # )
    response = openai.Completion.create(
        model=model,
        prompt=messages[0]['content'],
        temperature=temperature,
        max_tokens=max_tokens,
    ) 
    response_content = response.choices[0].text
    d = response['usage'].to_dict()
    d['content'] = response_content
    return d

def send_prompt(model, prompt, *args, **kwargs):
    prompt = prompt + "You need to provide a Python solution for the problem and include the necessary code to print out the result. Let's think step by step."
    messages = [
            {"role": "system","content": 'As an expert in graph machine learning proficient in Python, please provide solutions using libraries like NetworkX and PyTorch Geometric (PYG) for the following tasks related to graph theory and machine learning: link prediction ("when_link"), connectivity prediction ("when_connect"), transitive closure computation ("when_tclosure"), node classification ("what_node"), neighbor identification ("which_neighbor"), checking transitive closures ("check_tclosure"), checking transitive paths ("check_tpath"), finding transitive paths ("find_tpath"), and edge sorting ("sort_edge"). Include detailed Python code and explanations for each task.'},
            {"role": "user","content": prompt}
        ]
    if 'gpt' in model:
        if 'instruct' in model:
            response = send_to_chat_gpt_intruct(model,messages, *args, **kwargs)
        else:
            response = send_to_chat_gpt(model,messages, *args, **kwargs)
    else:
        response = send_to_chat(model,messages, *args, **kwargs)
    return response