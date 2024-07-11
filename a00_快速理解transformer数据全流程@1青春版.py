from arg import arg
import tiktoken
import math
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn

# ———————————————————加载并观测数据————————————————————————
with open('data/订单商品名称.csv', 'r', encoding='utf-8') as f:
    text = f.read()

print('数据原始类型：',type(text))

# 作为预训练模型，它的目的是学习海量的符号排列组合。
# 因此它需要的数据可以是任何符合语言学分布规律的文字。


# ————————————————————token化————————————————————————————
tokenizer = tiktoken.get_encoding("cl100k_base") # 实例化openai的tokenizer
token_text = tokenizer.encode(text) # 文字->token  [int,int,...]
token_torch = torch.tensor(token_text,dtype=torch.long,device=arg.device) # 转换成torch.tensor 1维格式

print('预训练数据总长度：',token_torch.shape)

# ——————————————————————数据分组————————————————————————————
split_line = int(len(token_torch)*0.9) # 训练测试数据集的分割线
train_data = token_torch[:split_line] # 0.9的数据用于训练
val_data = token_torch[split_line:] # 0.1数据用于验证

print('训练数据长度：',train_data.shape)
print('测试数据长度：',val_data.shape)

# ——————————————————————数据采样————————————————————————————————
sample_start_id = torch.randint(low=0,high=len(train_data)-arg.token_length,size=(arg.batch_size,)) # 生成每次采样的起始位置，由于采样最大长度的的设置，可能导致省略最后一些数据
x_batch_token = torch.stack([train_data[id:id+arg.token_length] for id in sample_start_id]) # 训练中每轮输入
y_batch_token = torch.stack([train_data[id+1:id+arg.token_length+1] for id in sample_start_id]) # 训练中每轮答案，往后多移动一个token。

print('批量采样的起始位置：',sample_start_id)

# 打印所有输入输出
for i in range(arg.batch_size):
    print(f'第{i+1}条x输入：',tokenizer.decode(x_batch_token[i,:].cpu().numpy()))
    print(f'第{i+1}条y输出：',tokenizer.decode(y_batch_token[i,:].cpu().numpy()))

print('第1步-批量采样数据输入形状：',x_batch_token.shape) # (batch_size,token_length)


# ——————————————————————vocab——————————————————————————————
arg.vocab = max(token_text)+1 # 词表中有多少词汇
# 因为是序列号，从0算起，而创建维度是你输入多少就创建多少维度，所以需要+1
print('vocab:',arg.vocab)

# ——————————————————————embedding————————————————————————————————
# 创建embedding权重
embedding_weights = nn.Embedding(arg.vocab,arg.d_model,device=arg.device)

# x的embedding
x_batch_embedding = embedding_weights(x_batch_token) # (batch_size,token_length,d_model)

print('embedding权重形状：',embedding_weights)
print('第2步-embedding的形状：',x_batch_embedding.shape)

# —————————————————————pos——————————————————————————————————————————
# 1创造一个巨大的位置编码矩阵
# 列数，和一些数值
position = torch.arange(0, arg.token_length) # (token_length) 以索引值作为位置信息
position = position.unsqueeze(1) # (token_length,1)

# 行数，和一些数值
x1 = torch.arange(0, arg.d_model, 2) # (0.5d_model)
b = -math.log(10000.0) / arg.d_model  # (0)
div_term = torch.exp(x1 * b) # (0.5d_model)

# 生成位置编码矩阵
pe = torch.zeros(arg.token_length, arg.d_model) # (token_length,d_model) 全0矩阵，用于存放运算结果
pe[:, 0::2] = torch.sin(position * div_term)  # (token_length,0.5d_model) 偶数 sin(索引值*变换矩阵)
pe[:, 1::2] = torch.cos(position * div_term)  # (token_length,0.5d_model) 奇数 cos(索引值*变化矩阵)
# pe (token_length,d_model)

# pe形状大小跟输入完全没有关系，完全由类的输入参数决定。
# 因为需要叠加，所以要剪裁至与输入大小相同的形状
pe = pe.unsqueeze(0)  # (1,token_length,d_model)

# 2截取至和输入一样的形状
p=pe[:, :x_batch_embedding.size(1), :x_batch_embedding.size(2)]

p=p.to(arg.device) # 转移到显存上

# 3叠加
x = x_batch_embedding + p # (B,T,D)

print('pos的形状：',p.shape)
print('第3步-加入位置编码pos后形状：',x.shape) # (B,T,D)

# ————————————————————————————乘以qkv权重——————————————————————————
# (列数，行数)
# (输入，输出)
wq = nn.Linear(arg.d_model,arg.d_model).to(arg.device)
wk = nn.Linear(arg.d_model,arg.d_model).to(arg.device)
wv = nn.Linear(arg.d_model,arg.d_model).to(arg.device)

q = wq(x)
k = wk(x)
v = wv(x)

print('第4步-qkv维度：',q.shape) # (B,T,D)

# ———————————————————————————多头————————————————————————————————
q = q.reshape(arg.batch_size,arg.token_length,arg.num_heads,arg.d_model//arg.num_heads).permute(0,2,1,3)
k = k.reshape(arg.batch_size,arg.token_length,arg.num_heads,arg.d_model//arg.num_heads).permute(0,2,1,3)
v = v.reshape(arg.batch_size,arg.token_length,arg.num_heads,arg.d_model//arg.num_heads).permute(0,2,1,3)

print('第5步-分头后qkv维度：',q.shape) # (B,H,T,D/H)

# ——————————————————————————softmax——————————————————————————————
# Q * K.T / √dk
s = q @ k.transpose(-2,-1) / math.sqrt(arg.d_model//arg.num_heads)
print('qk相乘形状：',s.shape) # (B,H,T,T)

# mask
mask = torch.full(
    (arg.token_length, arg.token_length),
    float("-inf")
)
mask = torch.triu(mask, diagonal=1) # 右下斜切，上三角全为-inf，下三角全为0

masked = s + mask
print('mask过后的样子:\n',pd.DataFrame(masked[0,0].detach().cpu().numpy()))




print('第6步：softmax转换为概率')

# softmax( Q * K.T / √dk )
p_attn = torch.softmax(masked, dim=-1) # (B,T,T)
print('softmax( Q * K.T / √dk )    注意力权重的形状:',p_attn.shape)

# attn = softmax( Q * K.T / √dk ) * V
attn = p_attn @ v  # (B,T,D/H)
print('softmax( Q * K.T / √dk )*V  注意力输出的形状:',attn.shape)

# ——————————————————————————合并多头————————————————————————————————
a = attn.transpose(1,2).reshape(arg.batch_size,-1,arg.d_model) # (B,T,D)
print('第7步-合并多头注意力的形状：',a.shape)

# —————————————————————————多头注意力输出————————————————————————————
wo = nn.Linear(arg.d_model,arg.d_model).to(arg.device)
attn_output = wo(a) # (B,T,D)

# ——————————————————————————残差连接1————————————————————————————————
# 残差
res1_in = attn_output + x
# 层归一化
layer_norm1 = nn.LayerNorm(arg.d_model).to(arg.device)
res1_out = layer_norm1(res1_in)

# ——————————————————————————前馈神经网络————————————————————————————————————
linear1 = nn.Linear(arg.d_model,arg.d_model * 4).to(arg.device)
feed1_out = linear1(res1_out)

feed1_out = nn.ReLU()(feed1_out) # 激活函数

linear2 = nn.Linear(arg.d_model * 4,arg.d_model).to(arg.device)
feed1_out = linear2(feed1_out)

# ——————————————————————————残差连接2————————————————————————————————————
# 残差
res2_in = feed1_out + res1_out
# 层归一化
layer_norm2 = nn.LayerNorm(arg.d_model).to(arg.device)
res2_out = layer_norm2(res2_in)

print('第9步-残差连接1+前馈神经网络+残差连接2 后的形状：',res2_out.shape) # (B,T,D)

# —————————————————————unembedding——————————————————————————
# unembedding是一大堆浮点数，数值高的是预测的
unembedding_weights = nn.Linear(arg.d_model,arg.vocab).to(arg.device)
output = unembedding_weights(res2_out)
print('unembedding权重的形状：',unembedding_weights.weight.shape)
print('第10步-unembedding后的形状：',output.shape)

# —————————————————————输出每一个词汇的概率——————————————————————————
logits = F.softmax(output,dim = -1) # 逻辑斯蒂回归，把任意实数映射到0-1区间
print('第11步-输出每一个词汇的概率，最后一个维度代表每一个字被预测的概率，从中挑选概率最大的')
print('输出概率logits的形状：',logits.shape)

max_indices = torch.argmax(logits, dim=2) # 第3维度中找到最大概率值的索引
print('每句话的token预测值：\n',max_indices)

print('预测第一行token：\n',max_indices[0,:].cpu().numpy()) # 第一行索引
print('预测第一行token的解码：\n',tokenizer.decode(max_indices[0,:].cpu().numpy())) # 根据索引解码出文字

# ——————————————————————计算loss交叉熵————————————————————————————————
# 改变成计算损失所需要的形状
B,T,D = logits.shape # 形状操作
logits_reshaped = logits.reshape((B*T,D)) #
y_batch_reshaped = y_batch_token.reshape(B*T) # 答案，用于计算损失函数
loss = F.cross_entropy(input=logits_reshaped,target=y_batch_reshaped)
print(loss)

# ——————————————————————计算梯度，更新权重————————————————————————————————
# 优化器，用于调整权重
optimizer = torch.optim.AdamW(

    # params 是需要更新的权重
    params= list(

            # ————————————embedding权重——————————
            embedding_weights.parameters())+

            # ————————————transformer权重—————————————
            # qkvo多头权重
            list(wq.parameters())+
            list(wk.parameters())+
            list(wv.parameters())+
            list(wo.parameters())+

            # 前馈神经网络权重
            list(layer_norm1.parameters())+

            list(linear1.parameters())+
            list(linear2.parameters())+

            list(layer_norm2.parameters())+

            # ———————————unembedding权重————————————

            list(unembedding_weights.parameters()),

    lr = 1e-4, # 学习率
)
loss.backward() # 使用学习率计算梯度
optimizer.step() # 更新权重