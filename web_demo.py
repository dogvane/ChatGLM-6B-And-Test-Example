import os
import platform
import pickle
from datetime import datetime

from transformers import AutoModel, AutoTokenizer
import gradio as gr


cache_dir = "G:/GPT/THUDM_chatglm-6b"

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

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir=cache_dir,  trust_remote_code=True)

quantizationBit = 4 # 量化模型的大小， 4 或者8 ，0 表示不进行量化
model = loadModel(quantizationBit)
print(f"\nload model ok: cost: { (datetime.now() - previous_time).total_seconds() } sec\n")

model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def check_history(wait_clean_item):
    length = sum(sum(len(s) for s in tpl) for tpl in wait_clean_item)
    print(f"list length:{len(wait_clean_item) } str length:{length}")
    while length > 2048:
        length -= sum(len(s) for s in wait_clean_item[0])
        del wait_clean_item[0]
        
        print(f'clear history less length: {sum(sum(len(s) for s in tpl) for tpl in wait_clean_item)}')
    
    return wait_clean_item


def predict(input, history=None):
    if history is None:
        history = []
    
    previous_time = datetime.now()
    response, history = model.chat(tokenizer, input, check_history(history))
    print(f"ChatGLM cost:{ (datetime.now() - previous_time).total_seconds() } sec\n")
    
    updates = []
    for query, response in history:
        updates.append(gr.update(visible=True, value="用户   ：" + query))
        updates.append(gr.update(visible=True, value="ChatGLM：" + response))
    if len(updates) < MAX_BOXES:
        updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
    return [history] + updates


with gr.Blocks() as demo:
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="提问："))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="回复："))

    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
        with gr.Column(scale=1):
            button = gr.Button("Generate")
    button.click(predict, [txt, state], [state] + text_boxes)
demo.queue().launch(share=True, inbrowser=True)
