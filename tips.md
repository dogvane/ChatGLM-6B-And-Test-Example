# 使用提示

## window 使用 python 虚拟环境安装

conda create -n chatgpt python=3.10 -y
conda activate chatgpt

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

cd GPT
git clone https://github.com/dogvane/ChatGLM-6B-And-Test-Example.git
cd ChatGLM-6B

pip install -r .\requirements.txt

## INT4 还是 INT8
gtx 1080 能跑 INT4 的 2048 token 的对话，INT8 能跑，但是不能使用history或者说，只能使用少量的token的history。


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
根据一些热心网友的回答，在4090下，能够达到chatgpt3的反馈速度。 1s~2s 给出一个结果。

