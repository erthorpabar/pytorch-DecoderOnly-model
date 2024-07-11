'''GPT'''

from dataclasses import dataclass
import torch
# ——————————————————模型超参数——————————————————————
# 调用方式：arg.参数名称
@dataclass
class arg:
    # ——————————————————硬件参数————————————————————
    device:str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ——————————————pre_train参数——————————————————————
    # 训练过程参数
    max_iters: int = 1000  # 训练最大次数
    eval_interval: int = 50  # 多少步评估一次
    eval_iters: int = 10  # 用多少数据做评估

    # 更新参数的策略
    learning_rate: float = 1e-3  # 学习率

    # 其他
    seed: int = 1337 # 随机种子

    # ————————————————model参数————————————————————————
    # embedding
    vocab: int = 100180  # 需要和tokenizer的识别数量一致 V

    # 多少层
    n_layer: int = 8  # 有多少transformer层

    # attention参数
    batch_size: int = 4  # 批次 B
    token_length: int = 64  # token长度 T
    d_model: int = 1024  # embedding维度 D
    num_heads: int = 8  # 头数 H

    groups: int = 4 # 仅分组q_kv_group会用到，需要是偶数，因为分组需要被整除

    # ffn参数
    n_multiple: int = 4 # 前馈神经网络ffn的缩放倍数

    # 其他
    dropout: float = 0.1  # 将输出的矩阵数值随机归零











