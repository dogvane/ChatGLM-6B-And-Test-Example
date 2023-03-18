import os
import platform
import pickle

from datetime import datetime
from transformers import AutoTokenizer, AutoModel

os_name = platform.system()
print("os: " + os_name)
previous_time = datetime.now()
cache_dir = "G:/GPT/THUDM_chatglm-6b";

print(f"begin load. {datetime.now()}")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", cache_dir=cache_dir, trust_remote_code=True)
print(f"\nload tokenizer ok. cose:{ (datetime.now() - previous_time).total_seconds()} sec\n")

def loadModel(quantizationBit):
    model = None
    filename = f'chatglm_{quantizationBit}bit.preq'
    filename = os.path.join(cache_dir, filename)

    if quantizationBit > 0:
        if os.path.exists(filename):
            print(f"find {filename}, begin load cache file.")
            with open(filename, 'rb') as f:
                return pickle.load(f).cuda()
                    
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
    return model

previous_time = datetime.now()
quantizationBit = 4 # 量化模型的大小， 4 或者8 ，0 表示不进行量化
model = loadModel(quantizationBit)
print(f"\nload model ok: cost: { (datetime.now() - previous_time).total_seconds() } sec\n")

model = model.eval()

history = []
print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")

def check_history(wait_clean_item):
    length = sum(sum(len(s) for s in tpl) for tpl in wait_clean_item)
    print(f"list length:{len(wait_clean_item) } str length:{length}")
    while length > 2048:
        length -= sum(len(s) for s in wait_clean_item[0])
        del wait_clean_item[0]
        
        print(f'clear history less length: {sum(sum(len(s) for s in tpl) for tpl in wait_clean_item)}')
    
    return wait_clean_item

while True:
    query = input("\n用户：")
    if query == "stop":
        break
    if query == "clear":
        history = []
        command = 'cls' if os_name == 'Windows' else 'clear'
        os.system(command)
        print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
        continue
    
    previous_time = datetime.now()
    response, history = model.chat(tokenizer, query, history=history)
    print(f"ChatGLM-6B：{response} \n: cost:{ (datetime.now() - previous_time).total_seconds() } sec\n")
    history = check_history(history)  
