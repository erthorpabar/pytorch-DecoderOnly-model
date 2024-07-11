import torch
from a01_model import GPT
from arg import arg

# ————————————————————加载模型——————————————————————————
checkpoint = torch.load('model/model.ckpt') # 加载ckpt

model = GPT(checkpoint['h_params']) # 实例化，并读取超参数
model.load_state_dict(state_dict=checkpoint['model_state_dict']) # 读取模型参数

model.eval() # 推理模式
model.to(arg.device)


# ———————————————————计算模型参数————————————————————————
# 打印模型结构
n_sum = 0
for pname,p in model.named_parameters():
    print('权重名称：',pname,'权重参数量：',p.numel())
    n_sum = n_sum + p.numel()
print('总参数量：',n_sum)

'''决定模型大小的参数量：不计算norm和偏置的话

0 embedding = unembedding = (vocab x d_model) x 2 
1 transformer的层数，n_layer 
2 单层内 没有分头的q,k,v,o = ( d_model x d_model) x 4 
3 单层内 ffn 的两个线性层 = (d_model x n_multiple x d_model) x 2 

不计算norm和偏置的参数总量 
= n_layer( 4( d_model x d_model) + 2(d_model x n_multiple x d_model)) + 2(vocab x d_model)
= 305831936
'''





