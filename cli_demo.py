import os
import platform
import signal
import pickle

from datetime import datetime
from transformers import AutoTokenizer, AutoModel

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
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

def build_prompt(history):
    prompt = ""
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

def main():
    quantizationBit = 4 # 量化模型的大小， 4 或者8 ，0 表示不进行量化
    tokenizer, model = loadModel(quantizationBit)

    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)

if __name__ == "__main__":
    main()
