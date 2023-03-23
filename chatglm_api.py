from fastapi import FastAPI
import os
import platform
import pickle

from datetime import datetime
from typing import List
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
print(f"os: {os_name} \nclear use command name: {clear_command}\n")

cache_dir = "./cache";

def loadModel(quantizationBit):
    previous_time = datetime.now()
    
    print(f"begin load tokenizer. {datetime.now()}")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", cache_dir=cache_dir, trust_remote_code=True)
    print(f"\nload tokenizer ok. cost:{ (datetime.now() - previous_time).total_seconds()} sec\n")

    model = None
    filename = f'chatglm_{quantizationBit}bit.preq'
    filename = os.path.join(cache_dir, filename)

    previous_time = datetime.now()

    if quantizationBit > 0:
        if os.path.exists(filename):
            print(f"find {filename}, begin load cache file.")
            with open(filename, 'rb') as f:
                model = pickle.load(f).cuda()
                    
    if model is None:
        if quantizationBit > 0:
            model = AutoModel.from_pretrained("THUDM/chatglm-6b",  
                                    cache_dir=cache_dir, 
                                    trust_remote_code=True
                                    ).half().quantize(quantizationBit).cuda()
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        else:
            model = AutoModel.from_pretrained("THUDM/chatglm-6b",  
                                cache_dir=cache_dir, 
                                trust_remote_code=True
                                ).half().cuda()
    print(f"\nload model ok: cost: { (datetime.now() - previous_time).total_seconds() } sec\n")

    model = model.eval()
    return tokenizer, model

model = None
tokenizer = None

quantizationBit = 4 # 量化模型的大小， 4 或者8 ，0 表示不进行量化
tokenizer, model = loadModel(quantizationBit)

@app.get("/")
def hello():
    return {"message": "Hello ChatGLM API!"}

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
    