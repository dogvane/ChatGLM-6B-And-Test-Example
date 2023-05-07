# 使用提示

## window 使用 python 虚拟环境安装

```
conda create -n chatgpt python=3.10 -y
conda activate chatgpt

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

// 目前是安装到2.0版本了
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cd GPT
git clone https://github.com/dogvane/ChatGLM-6B-And-Test-Example.git
cd ChatGLM-6B

pip install -r .\requirements.txt
```

或者到b站秋叶找一键安装包

https://www.bilibili.com/video/BV1E24y1u7Go



## 量化选 INT4 还是 INT8
gtx 1080 能完整的运行 INT4 的 2048 token对话。
INT8 也能跑，只能设置少量的上下文token，不适合对话场景。


## 爆显存的原因

```
Token indices sequence length is longer than the specified maximum sequence length for this model (2077 > 2048). Running this sequence through the model will result in indexing errors
Input length of input_ids is 2077, but `max_length` is set to 2048. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.
```

如果上面的问题，检查代码

```
 response, history = model.chat(tokenizer, query, history=history)
```

看看history是否过长，过长需要删除 history 里的一些上下文

## 性能

使用GTX 1080显卡，在INT4下，每次查询基本上都是30s起步，根据返回结果的长度，通常会到60s~120s之间。


## 加速启动优化

参考 https://github.com/THUDM/ChatGLM-6B/issues/143 里说的，当前的版本可以做到启动从 300s 变成 10s
