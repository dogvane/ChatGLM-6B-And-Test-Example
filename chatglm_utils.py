
from fastapi import FastAPI
import os
import platform
import pickle

from datetime import datetime
from typing import List
from transformers import AutoTokenizer, AutoModel

# cache_dir = "./cache";

def loadModel(quantizationBit = 0, cache_dir = './cache'):
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
