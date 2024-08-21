'''GPT'''
# ——————————————————导入包——————————————————————————
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from arg import arg

'''层级结构
GPT input(B,T)
-embedding_weights -> x_batch_embedding
-pos -> pe
-x = x_batch_embedding + pe
-transformer_block -> res2_out
    -norm1
    -qkv  wo( cat( softmax( mask( Q @ K.T / √dk (Q=wq@x K=wk@x V=wv@x

    -norm2
    -ffn  缩小n倍( 放大n倍
-out_norm -> output
-unembedding_weights -> output_token_value

'''
# ——————————————————GPT————————————————————————————
class GPT(nn.Module):
    def __init__(self, arg):  # self 是一个包含定义好模型名称，和参数的类
        super().__init__()
        self.arg = arg  # 输入所有参数
        self.token_length = arg.token_length
        self.d_model = arg.d_model
        self.n_layer = arg.n_layer
        self.num_heads = arg.num_heads
        self.dropout = arg.dropout
        self.vocab = arg.vocab
        self.device = arg.device

        # ————————————————创造模型参数————————————————
        self.embedding_weights = nn.Embedding(self.vocab, self.d_model)

        self.transformer_block = nn.Sequential(
            *[Transformer_Block(self.arg) for _ in range(self.n_layer)]
        )

        # 两种norm可选
        # self.out_norm = nn.LayerNorm(self.d_model)
        # self.out_norm = LayerNorm(self.d_model) # 手动定义的
        self.out_norm = RMSNorm(self.d_model)

        self.unembedding_weights = nn.Linear(self.d_model, self.vocab)

        # ————————————————共享参数———————————————————————————
        # 实际测试效果明显不太好。
        # 作用是共享参数，在计算参数量的时候算入embedding_weight中。可减少参数量。
        # self.embedding_weights.weight = self.unembedding_weights.weight


        # ——————————————调整模型权重初始分布————————————————————
        # 实际测试效果明显不太好。使用torch默认的分布更好。
        # for module in self.modules():
        #     # 如果是Linear
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 重置权重，正态分布
        #         if module.bias is not None:  # 如果有偏置
        #             torch.nn.init.zeros_(module.bias)  # 则变为0
        #     # 如果是Embedding
        #     elif isinstance(module, nn.Embedding):
        #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 重置权重，正态分布

        # —————————————————pos位置编码矩阵—————————————————————
        # 1创造一个巨大的位置编码矩阵
        # 列数，和一些数值
        position = torch.arange(0, self.token_length, dtype=torch.float)  # (token_length) 以索引值作为位置信息
        position = position.unsqueeze(1)  # (token_length,1)

        # 行数，和一些数值
        x1 = torch.arange(0, self.d_model, 2).float()  # (0.5d_model)
        b = -math.log(10000.0) / self.d_model  # (0)
        div_term = torch.exp(x1 * b)  # (0.5d_model)

        # 生成位置编码矩阵
        pe = torch.zeros(self.token_length, self.d_model, device=self.device)  # (token_length,d_model) 全0矩阵，用于存放运算结果
        pe[:, 0::2] = torch.sin(position * div_term)  # (token_length,0.5d_model) 偶数 sin(索引值*变换矩阵)
        pe[:, 1::2] = torch.cos(position * div_term)  # (token_length,0.5d_model) 奇数 cos(索引值*变化矩阵)
        # pe (token_length,d_model)

        # pe形状大小跟输入完全没有关系，完全由类的输入参数决定。
        # 因为需要叠加，所以要剪裁至与输入大小相同的形状
        pe = pe.unsqueeze(0)  # (1,token_length,d_model)

        # 其他操作
        self.register_buffer('pe', pe)  # 注册成buffer，1不更新参数，2可保存后加载调用

        # pe是一个固定的位置编码矩阵，无论输入的是什么长度的矩阵，只需要剪裁pe然后添加即可。需确保pe长度大于输入的矩阵。

    def forward(self, x_batch_token, y_batch_token=None):
        # x_batch_token(B,T)
        # B, T = x_batch_token.shape

        # ——————————————embedding————————————————
        x_batch_embedding = self.embedding_weights(x_batch_token)  # (B,T,D)

        # ————————————————pos————————————————————
        # 2截取至和输入一样的形状
        p = self.pe[:, :x_batch_embedding.size(1), :x_batch_embedding.size(2)].to(self.device)  # (1,T,D)

        # ————————————————合并输入—————————————————
        x = x_batch_embedding + p  # (B,T,D)

        # ———————————transformer_block——————————————
        res2_out = self.transformer_block(x)  # (B,T,D)

        # —————————————layernorm归一化———————————————
        output = self.out_norm(res2_out)  # (B,T,D)

        # ————————————unembedding线性层————————————————
        output_token_value = self.unembedding_weights(output)  # (B,T,V)

        # ——————————————转换为百分比概率—————————————————
        # logits = F.softmax(output_token_value, dim=-1)

        # ——————————————计算交叉熵损失——————————————————
        if y_batch_token is not None:  # 即开启反向传播
            B, T, V = output_token_value.shape
            output_reshaped = output_token_value.reshape((B * T, V))
            y_batch_reshaped = y_batch_token.reshape(B * T)
            loss = F.cross_entropy(input=output_reshaped, target=y_batch_reshaped)
        else:
            loss = None

        return output_token_value, loss  # output_token_value 还需要转化成百分比

    # 每次生成一个token，生成n次
    # x_batch_token (B,T)
    def generate(self, x_batch_token, max_new_tokens=100, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            x_crop = x_batch_token[:, -arg.token_length:]  # 每次截取最新的最大长度值
            output_token_value, loss = self.forward(x_crop)
            # (B,T,V)

            output_token_value = output_token_value[:, -1, :] / temperature  # 裁剪过后

            if top_k is not None:
                v, _ = torch.topk(output_token_value, min(top_k, output_token_value.size(-1)))
                output_token_value[output_token_value < v[:, [-1]]] = -float('Inf')

            logits = F.softmax(input=output_token_value, dim=-1)  # 转换为概率

            # 采样生成1个样本
            next_token = torch.multinomial(logits, num_samples=1)  # 采样返回1个样本

            x_batch_token = torch.cat((x_batch_token, next_token), dim=1)

        return x_batch_token

# ————————————————————transformer层——————————————————————————
class Transformer_Block(nn.Module): # 输入内容为 embedding + pos  x(B,T,D)
    def __init__(self,arg):
        super().__init__()

        self.norm1 = nn.LayerNorm(arg.d_model)
        self.mha = qkv_HeadAttention(arg) # qkv多头
        # self.mha = q_HeadAttention(arg) # 只有q多头
        # self.mha = q_kv_group_HeadAttention(arg) # q_kv分组多头

        self.norm2 = nn.LayerNorm(arg.d_model)
        self.ffn = FeedForwardNetwork(arg)
        # self.ffn = llama_FeedForwardNetwork(arg)

    def forward(self,x): # x(B,T,D)
        # 残差链接1： norm1 + MultiHeadAttention(多头) + dropout(随机归零)
        res1_in = x  # 输入
        attn_output = self.mha(self.norm1(res1_in))  # 残差1 + 多头
        res1_out = res1_in + attn_output

        # 残差链接2： norm2 + FeedForwardNetwork(前馈缩放) + dropout(随机归零)
        res2_in = res1_out  # 输入
        ffn_output = self.ffn(self.norm2(res2_in))  # 残差2 + 前馈缩放
        res2_out = res2_in + ffn_output  # (B,T,D)

        return res2_out

# —————————————————MultiHeadAttention多头注意力————————————————
# mha
class qkv_HeadAttention(nn.Module): # 输入为分头后的embedding + pos
    def __init__(self,arg):
        super().__init__()
        self.arg = arg # 输入所有参数
        self.d_model = arg.d_model
        self.num_heads = arg.num_heads
        self.token_length = arg.token_length
        self.dropout = arg.dropout

        self.wq = nn.ModuleList([nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads)])
        self.wk = nn.ModuleList([nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads)])
        self.wv = nn.ModuleList([nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads)])

        self.wo = nn.Linear(self.d_model, self.d_model)

        self.register_buffer(
            'mask',
            torch.triu(  # 右下斜切，上三角全为-inf，下三角全为0
                torch.full((arg.token_length, arg.token_length), float("-inf")),  # 全-inf的矩阵
                diagonal=1
            )
        )

        self.Dropout = nn.Dropout(self.dropout)

    def forward(self,x): # x(B,T,D)

        # ————————————————注意力机制——————————————————————
        attn_outs = []
        for i in range(self.num_heads):
            # 分头注意力计算
            q = self.wq[i](x) # (B,T,D/H)
            k = self.wk[i](x) # (B,T,D/H)
            v = self.wv[i](x) # (B,T,D/H)

            # Q @ K.T / √dk
            s = q @ k.transpose(-2, -1) / math.sqrt(self.d_model // self.num_heads)  # s(B,T,T)

            # mask
            B, T, D = x.shape
            masked = s + self.mask[:T, :T]  # 加入mask信息

            # softmax( Q @ K.T / √dk )
            p_attn = F.softmax(masked, dim=-1)  # p_attn(B,T,T)

            # dropout(可选)
            p_attn = self.Dropout(p_attn)

            # attn = softmax( Q @ K.T / √dk ) @ V
            attn = p_attn @ v  # attn(B,T,D/H)

            # 将分头结果保存到列表中
            attn_outs.append(attn)

        # 合并多头
        a = torch.cat(attn_outs,dim = -1) # a(B,T,D)

        # 计算多头注意力输出
        attn_output = self.wo(a)  # attn_output(B,T,D)

        # dropout(可选)
        attn_output = self.Dropout(attn_output)

        return attn_output

# mqa 效果差一点，但是节省参数量
class q_HeadAttention(nn.Module): # 输入为分头后的embedding + pos
    def __init__(self,arg):
        super().__init__()
        self.arg = arg # 输入所有参数
        self.d_model = arg.d_model
        self.num_heads = arg.num_heads
        self.token_length = arg.token_length
        self.dropout = arg.dropout

        self.wq = nn.ModuleList([nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads)])
        self.wk = nn.Linear(self.d_model, self.d_model // self.num_heads)
        self.wv = nn.Linear(self.d_model, self.d_model // self.num_heads)

        self.wo = nn.Linear(self.d_model, self.d_model)

        self.register_buffer(
            'mask',
            torch.triu(  # 右下斜切，上三角全为-inf，下三角全为0
                torch.full((arg.token_length, arg.token_length), float("-inf")),  # 全-inf的矩阵
                diagonal=1
            )
        )

        self.Dropout = nn.Dropout(self.dropout)

    def forward(self,x): # x(B,T,D)

        # ————————————————注意力机制——————————————————————
        attn_outs = []
        for i in range(self.num_heads):
            # 多头注意力计算，只有q改变，其他不变，可减少一定参数量
            q = self.wq[i](x) # (B,T,D/H)
            k = self.wk(x) # (B,T,D/H)
            v = self.wv(x) # (B,T,D/H)

            # Q @ K.T / √dk
            s = q @ k.transpose(-2, -1) / math.sqrt(self.d_model // self.num_heads)  # s(B,T,T)

            # mask
            B, T, D = x.shape
            masked = s + self.mask[:T, :T]  # 加入mask信息

            # softmax( Q @ K.T / √dk )
            p_attn = F.softmax(masked, dim=-1)  # p_attn(B,T,T)

            # dropout(可选)
            p_attn = self.Dropout(p_attn)

            # attn = softmax( Q @ K.T / √dk ) @ V
            attn = p_attn @ v  # attn(B,T,D/H)

            # 将分头结果保存到列表中
            attn_outs.append(attn)

        # 合并多头
        a = torch.cat(attn_outs,dim = -1) # a(B,T,D)

        # 计算多头注意力输出
        attn_output = self.wo(a)  # attn_output(B,T,D)

        # dropout(可选)
        attn_output = self.Dropout(attn_output)

        return attn_output

# gqa
class q_kv_group_HeadAttention(nn.Module): # 输入为分头后的embedding + pos
    def __init__(self,arg):
        super().__init__()
        self.arg = arg # 输入所有参数
        self.d_model = arg.d_model
        self.num_heads = arg.num_heads
        self.token_length = arg.token_length
        self.groups = arg.groups
        self.dropout = arg.dropout

        self.wq = nn.ModuleList([nn.Linear(self.d_model, (self.d_model // self.num_heads)//self.groups ) for _ in range(self.num_heads*self.groups)])
        self.wk = nn.ModuleList([nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads)])
        self.wv = nn.ModuleList([nn.Linear(self.d_model, self.d_model // self.num_heads) for _ in range(self.num_heads)])

        self.wo = nn.Linear(self.d_model, self.d_model)

        self.register_buffer(
            'mask',
            torch.triu(  # 右下斜切，上三角全为-inf，下三角全为0
                torch.full((arg.token_length, arg.token_length), float("-inf")),  # 全-inf的矩阵
                diagonal=1
            )
        )

        self.Dropout = nn.Dropout(self.dropout)

    def forward(self,x): # x(B,T,D)

        # ————————————————注意力机制——————————————————————
        # 运算wq(x)
        q_group_outs = [] # (B,T,(D/H)/g )
        for i in range(self.num_heads*self.groups):
            q_out = self.wq[i](x)
            q_group_outs.append(q_out)

        # 每g个合并q
        q_cat_outs = [] # (B,T,(D/H))
        for i in range(0, self.num_heads*self.groups, self.groups):
            q_cat = torch.cat(
                q_group_outs[i:i+self.groups],
                dim = -1
            )
            q_cat_outs.append(q_cat)

        attn_outs = []
        for i in range(self.num_heads):

            q = q_cat_outs[i] # (B,T,(D/H))
            k = self.wk[i](x) # (B,T,(D/H))
            v = self.wv[i](x) # (B,T,(D/H))

            # Q @ K.T / √dk
            s = q @ k.transpose(-2, -1) / math.sqrt(self.d_model // self.num_heads)  # s(B,T,T)

            # mask
            B, T, D = x.shape
            masked = s + self.mask[:T, :T]  # 加入mask信息

            # softmax( Q @ K.T / √dk )
            p_attn = F.softmax(masked, dim=-1)  # p_attn(B,T,T)

            # dropout(可选)
            p_attn = self.Dropout(p_attn)

            # attn = softmax( Q @ K.T / √dk ) @ V
            attn = p_attn @ v  # attn(B,T,D/H)

            # 将分头结果保存到列表中
            attn_outs.append(attn)

        # 合并多头
        a = torch.cat(attn_outs,dim = -1) # a(B,T,D)

        # 计算多头注意力输出
        attn_output = self.wo(a)  # attn_output(B,T,D)

        # dropout(可选)
        attn_output = self.Dropout(attn_output)

        return attn_output # attn_output(B,T,D)
# ——————————————————————feedforward——————————————————————————
class FeedForwardNetwork(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.d_model = arg.d_model
        self.n_multiple = arg.n_multiple
        self.dropout = arg.dropout

        self.ffn = nn.Sequential( # 按顺序执行
            nn.Linear(self.d_model, self.d_model * self.n_multiple),
            nn.ReLU(),
            nn.Linear(self.d_model * self.n_multiple, self.d_model),

            nn.Dropout(self.dropout)
        )

    def forward(self,x): # x (B,T,D)
        return self.ffn(x)  #  (B,T,D)

class llama_FeedForwardNetwork(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.d_model = arg.d_model
        self.n_multiple = arg.n_multiple

        self.w1 = nn.Linear(self.d_model, self.d_model * self.n_multiple, bias=False)
        self.w2 = nn.Linear(self.d_model, self.d_model * self.n_multiple, bias=False)

        self.w3 = nn.Linear(self.d_model * self.n_multiple, self.d_model, bias=False)

    def forward(self,x): # x (B,T,D)
        return self.w3(F.silu(self.w1(x)) * self.w2(x)) # *表示矩阵对位相乘
        # out (B,T,D)

# ————————————————————norm————————————————————————————
# 不改变矩阵形状，只对最后一个维度归一化
# 实例化LayerNorm(d_model)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        # Create two learnable parameters for normalization
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Calculate the mean and standard deviation
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)

        # Normalize the batch of tensors
        x_normalized = (x - mean) / (std + self.eps)

        # Scale and shift
        return self.gamma * x_normalized + self.beta
# 使用方法
# input = torch.randn(3, 5)
# layer_norm = nn.LayerNorm(input.size(-1))
# output = layer_norm(input)
# print(output.shape)
# 形状为(3, 5)

# 不改变矩阵形状，只对最后一个维度归一化
# 实例化RMSNorm(d_model)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
# 使用方法
# input = torch.randn(3, 5)
# rms_norm = RMSNorm(dim=input.size(-1))
# output = rms_norm(input)
# print(output.shape)
# 形状为(3, 5)

