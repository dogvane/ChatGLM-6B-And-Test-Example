from fastapi import FastAPI
import os
import platform
import pickle
import chatglm_utils

from datetime import datetime
from typing import List
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
print(f"os: {os_name} \nclear use command name: {clear_command}\n")

model = None
tokenizer = None

quantizationBit = 4 # 量化模型的大小， 4 或者8 ，0 表示不进行量化
tokenizer, model = chatglm_utils.loadModel(quantizationBit)

@app.get("/")
def hello():
    return {"message": "Hello ChatGLM API!"}

@app.route('/translate', methods=['GET', 'POST'])
def pred_chat(user_msg: str):
    print(f'model {model} {tokenizer}')
    response, history = model.chat(tokenizer, user_msg, [('翻译', "")])
    return {"response": response,
            "history": history}
    
@app.get("/predict")
def pred_chat(user_msg: str):
    print(f'model {model} {tokenizer}')
    response, history = model.chat(tokenizer, user_msg, [])
    return {"response": response,
            "history": history}
    
@app.post("/predict")
def pred_chat(user_msg: str,
              history: List[List[str]]):
    response, history = model.chat(tokenizer, user_msg, history)
    return {"response": response,
            "history": history}
    