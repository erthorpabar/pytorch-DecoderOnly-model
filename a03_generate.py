# ——————————————————导入包——————————————————————————
import torch
import tiktoken
from a01_model import GPT
from arg import arg

# ————————————————————加载模型——————————————————————————
checkpoint = torch.load('model/model.ckpt') # 加载ckpt

model = GPT(checkpoint['h_params']) # 实例化，并读取超参数
model.load_state_dict(state_dict=checkpoint['model_state_dict']) # 读取模型参数

model.eval() # 推理模式
model.to(arg.device)

# ———————————————————加载tokenizer——————————————————————
tokenizer = tiktoken.get_encoding("cl100k_base")


# ———————————————————输入————————————————————————————————
text = "农夫山泉 "

# ———————————————————输入做数据处理————————————————————————
token_text = tokenizer.encode(text) # str -> token
x_token_torch = (torch.tensor(token_text,dtype=torch.long,device=arg.device)[None, ...])
# 用none在最前增加一个维度

# ———————————————————推理————————————————————————————————
with torch.no_grad():
    y_token_torch = model.generate(x_token_torch,max_new_tokens=64,temperature=1.0, top_k=None)
    print(tokenizer.decode(y_token_torch[0].tolist()))


