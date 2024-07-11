# ——————————————————————导入包————————————————————————
import torch
import torch.nn.functional as F

# ———————————————————0 输入数据处理————————————————————————

# 原始输入，[你好，啊]

# 输入：文字->通过tokenizer->embedding
em =   [[0.11, 0.21, 0.31, 0.41],
        [0.21, 0.31, 0.41, 0.51],
        [0.31, 0.41, 0.51, 0.61],
        [0.41, 0.51, 0.61, 0.71]]
em = torch.tensor(em)



# embedding -> 位置编码 -> position embedding
pos =  [[0.01,0.01,0.01,0.01],
        [0.02,0.02,0.02,0.02],
        [0.03,0.03,0.03,0.03],
        [0.04,0.04,0.04,0.04]]
pos = torch.tensor(pos)

# encode = embedding + position embedding
encode=em+pos

# ———————————————————1 transformer过程1-注意力机制多头注意力————————————————————————————
# 多头注意力权重
wQ1=   [[0.3200, 0.4280, ],
        [0.4300, 0.5820, ],]

wK1=   [[0.2984, 0.4064, ],
        [0.3996, 0.5516, ],]

wV1=   [[0.3416, 0.4496, ],
        [0.4604, 0.6124, ],]

wQ2=   [[0.6440, 0.7520, ],
        [0.8860, 1.0380, ],]

wK2=   [[0.6440, 0.7520, ],
        [0.8860, 1.0380, ],]

wV2=   [[0.6440, 0.7520, ],
        [0.8860, 1.0380, ],]

wQ1 = torch.tensor(wQ1)
wK1 = torch.tensor(wK1)
wV1 = torch.tensor(wV1)

wQ2 = torch.tensor(wQ2)
wK2 = torch.tensor(wK2)
wV2 = torch.tensor(wV2)


# 多头(4,2)
encode1 = encode[:,:2]
encode2 = encode[:,2:]

# 多头注意力运算
Q1=encode1 @ wQ1
K1=encode1 @ wK1
V1=encode1 @ wV1

Q2=encode2 @ wQ2
K2=encode2 @ wK2
V2=encode2 @ wV2


# 计算相似度
# z = softmax(Q * K.T)/√dk * V
raw_attention_map1 = Q1 @ K1.T # 相关性 = Q *K.T
raw_attention_map1 = raw_attention_map1/8 # 除以√dk，这里直接指定为8
softmax = F.softmax(raw_attention_map1, dim=1) # softmax归一化
Z1= softmax @ V1

raw_attention_map2 = Q2 @ K2.T
raw_attention_map2 = raw_attention_map2/8
softmax = F.softmax(raw_attention_map2, dim=1)
Z2= softmax @ V2

# 合并
# 多注意力矩阵变为一个与encode形状相同的矩阵
z = torch.cat((Z1, Z2), dim=1)

wo=[[0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4],
    [0.1,0.2,0.3,0.4],]
wo = torch.tensor(wo)

zz = z @ wo

# —————————————————1 transformer过程2-前馈神经网络————————————————————

# layerNorm(x1)
x1_in = zz + encode

w1=    [[0.2,0.3,0.5,0.4],
        [0.2,0.3,0.5,0.4],
        [0.2,0.3,0.5,0.4],
        [0.2,0.3,0.5,0.4]]
w1 = torch.tensor(w1)

x1_out = x1_in @ w1


# 前馈神经网络
# 对维度进行缩放，放大4倍再缩小4倍


# layerNorm(x2)
x2_in = x1_out + x1_in

w2=    [[0.2,0.3,0.5,0.4],
        [0.2,0.3,0.5,0.4],
        [0.2,0.3,0.5,0.4],
        [0.2,0.3,0.5,0.4]]
w2 = torch.tensor(w2)

x2_out = x2_in @ w2
# —————————————————— 单层encoder完——————————————————————————
print(0)






