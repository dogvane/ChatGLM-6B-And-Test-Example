import os
import platform
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

os_name = platform.system()
print("os: " + os_name)
previous_time = datetime.now()

print(f"begin load. {datetime.now()}")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", cache_dir="G:/GPT/THUDM_chatglm-6b", trust_remote_code=True)
print(f"\nload tokenizer ok. cose:{ (datetime.now() - previous_time).total_seconds()} sec\n")

previous_time = datetime.now()

model = AutoModel.from_pretrained("THUDM/chatglm-6b",  
                                  cache_dir="G:/GPT/THUDM_chatglm-6b", 
                                  trust_remote_code=True
                                  ).half().quantize(4).cuda()
# 进行 2 至 3 轮对话后，8-bit 量化下 GPU 显存占用约为 10GB，4-bit 量化下仅需 6GB 占用。随着对话轮数的增多，对应消耗显存也随之增长，由于采用了相对位置编码，理论上 ChatGLM-6B 支持无限长的 context-length，但总长度超过 2048（训练长度）后性能会逐渐下降。
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
