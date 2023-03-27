import os
import chatglm_utils

from datetime import datetime
from typing import List

tokenizer = None
model = None

def read_and_translate_file(file_name):
    
    global model,tokenizer
    if  model is None:
        quantizationBit = 4 # 量化模型的大小， 4 或者8 ，0 表示不进行量化
        tokenizer, model = chatglm_utils.loadModel(quantizationBit)
    
    filename, file_extension = os.path.splitext(file_name)
    # 创建新的输出文件名
    output_file_name = filename + "_translate" + file_extension
    
    previous_time = datetime.now()
    lineCount = 0
    strCount = 0
    
    with open(file_name, 'r', encoding='utf-8') as file:
        with open(output_file_name, 'w', encoding='utf-8') as output_file:
            lines = file.readlines()
            
            # 遍历每一行内容
            for line in lines:
                if len(line.strip()) == 0:
                    output_file.write('\n')
                    continue
                
                lineCount += 1
                strCount += len(line)
                
                response, history =  model.chat(tokenizer, line, [('翻译：', '')])
                print(line)
                print(response)
                
                output_file.write(response + '\n')
                
    print(f"翻译语句数：{lineCount} 字符数量:{strCount} 总用时:{ (datetime.now() - previous_time).total_seconds() } 秒")

                

read_and_translate_file('./translate_source.txt')