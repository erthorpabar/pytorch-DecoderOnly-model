# ——————————————————导入包————————————————————————————
from arg import arg
import tiktoken
from a01_model import GPT
import torch.cuda.amp as amp

import torch

# ——————————————————固定种子————————————————————————————
torch.manual_seed(arg.seed)

# ———————————————————加载并观测数据————————————————————————
with open('data/订单商品名称.csv', 'r', encoding='utf-8') as f:
    text = f.read()

print('数据原始类型：',type(text))

# 作为预训练模型，它的目的是学习海量的符号排列组合。
# 因此它需要的数据可以是任何符合语言学分布规律的文字。
# 输入任意连续文本

# ————————————————————token化————————————————————————————
tokenizer = tiktoken.get_encoding("cl100k_base") # 实例化openai的tokenizer
token_text = tokenizer.encode(text) # 文字->token  [int,int,...]
token_torch = torch.tensor(token_text,dtype=torch.long,device=arg.device) # 转换成torch.tensor 1维格式

print('预训练数据总长度：',token_torch.shape)

# ——————————————————————计算vocab——————————————————————————————
arg.vocab = max(token_text)+1 # 词表中有多少词汇
# 因为是序列号，从0算起，而创建维度是你输入多少就创建多少维度，所以需要+1
print('vocab需要手动修改到arg文件:',arg.vocab)
# ——————————————————————数据分组————————————————————————————
split_line = int(len(token_torch)*0.9) # 训练测试数据集的分割线
train_data = token_torch[:split_line] # 0.9的数据用于训练
val_data = token_torch[split_line:] # 0.1数据用于验证

print('训练数据长度：',train_data.shape)
print('测试数据长度：',val_data.shape)

# ——————————————————————数据采样————————————————————————————————
def get_batch(split):
    if split =='train':
        data = train_data
    else:
        data = val_data

    sample_start_id = torch.randint(low=0, high=len(train_data) - arg.token_length, size=(arg.batch_size,)) # 生成每次采样的起始位置，由于采样最大长度的的设置，可能导致省略最后一些数据。
    x_batch_token = torch.stack([train_data[id:id+arg.token_length] for id in sample_start_id]) # 训练中每轮输入
    y_batch_token = torch.stack([train_data[id + 1:id + arg.token_length + 1] for id in sample_start_id]) # 训练中每轮答案，往后多移动一个token。
    return x_batch_token , y_batch_token

# ——————————————————————实例化——————————————————————————————————
model = GPT(arg).to(arg.device)

# 打印模型结构
n_sum = 0
for pname,p in model.named_parameters():
    print('权重名称：',pname,'权重参数量：',p.numel())
    n_sum = n_sum + p.numel()
print('总参数量：',n_sum)

# 打印参数总量
total_params = 0
for p in model.parameters():
    if p.requires_grad:
        total_params += p.numel()
print('参数总量:',total_params)










# ——————————————————————更新参数策略—————————————————————————————
optimizer = torch.optim.AdamW(
    params = model.parameters(), # 更新哪些参数
    lr = arg.learning_rate,
)

# ——————————————————————计算loss————————————————————————————————
@torch.no_grad() # 不累计梯度
def estimate_loss():
    model.eval()  # 推理模式不更新参数

    out = {}
    for split in ['train', 'valid']:
        losses = torch.zeros(arg.eval_iters)  # 全0的1维度张量
        for k in range(arg.eval_iters):
            x_batch_token, y_batch_token = get_batch('train')
            output_token_value, loss = model(x_batch_token, y_batch_token)
            losses[k] = loss.item() # 从张量中获取数值
        out[split] = losses.mean()  # 求平均

    model.train()  # 改回训练模式

    return out  # {'train':在训练集上的损失值,'valid':在验证集上的损失值,}

# ——————————————————————开始训练————————————————————————————————
print('训练硬件：',arg.device)

for step in range(arg.max_iters):

    # ————————————在训练n步时候打印训练结果——————————————————————
    if step % arg.eval_interval == 0 or step == arg.max_iters - 1:  # 每隔多少步
        losses = estimate_loss() # 计算loss，手动计算
        print('训练次数：', step,'训练集损失：', losses['train'],'验证集损失：', losses['valid'])

    # ———————————————————————训练————————————————————————————
    x_batch_token, y_batch_token = get_batch('train') # 从训练集上抽取一批输入进去
    output_token_value, loss = model(x_batch_token, y_batch_token) # pytorch计算

    optimizer.zero_grad(set_to_none=True)  # 清零上一步梯度

    loss.backward()  # 计算当前梯度
    optimizer.step()  # 更新权重






# ——————————————————————保存模型——————————————————————————————
torch.save(
    {
        'model_state_dict':model.state_dict(),
        'h_params': arg
     },
    'model/model.ckpt', # 保存地址
)














