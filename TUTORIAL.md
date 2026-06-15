# RL with Cold Start —— 多模态推理训练完整教程

> 本教程基于 [Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start](https://arxiv.org/pdf/2505.22334)（arXiv:2505.22334）的实现，使用 Qwen2.5-VL 作为基座模型，分两阶段（SFT cold start → GRPO）训练多模态推理能力。
>
> 阅读顺序：第一章建立背景 → 第二章理解模型 → 第三章掌握 SFT → 第四章理解 GRPO → 第五章实战 → 第六章排错与调优。

---

## 📚 目录

- [第一章 整体框架与训练范式](#第一章-整体框架与训练范式)
- [第二章 Qwen2.5-VL：模型如何"看图"](#第二章-qwen25-vl模型如何看图)
- [第三章 SFT 阶段：让模型学会推理格式](#第三章-sft-阶段让模型学会推理格式)
- [第四章 GRPO 阶段：用强化学习强化推理能力](#第四章-grpo-阶段用强化学习强化推理能力)
- [第五章 实战手册：从环境配置到训练完成](#第五章-实战手册从环境配置到训练完成)
- [第六章 排错与调优指南](#第六章-排错与调优指南)
- [附录 A：项目代码地图](#附录-a项目代码地图)
- [附录 B：超参速查表](#附录-b超参速查表)

---

## 第一章 整体框架与训练范式

### 1.1 项目目标

让多模态大模型（MLLM）在**几何题、图表理解、科学问答**等需要"看图 + 多步推理"的任务上有更强能力。

衡量标准（论文 benchmark）：
- **MathVista**: 66.3% → **73.4%** (7B)
- **We-Math**: 62.9% → **70.4%** (7B)
- 3B 模型能达到很多 7B 模型的水平

### 1.2 两阶段训练范式

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Qwen2.5-VL-3B/7B (基础模型)                                     │
│         │                                                       │
│         │ 阶段 1: SFT (Supervised Fine-Tuning) Cold Start       │
│         │  • 51K 条 (query, response) 数据                       │
│         │  • Teacher: Qwen2.5-VL-32B 蒸馏的高质量 CoT            │
│         │  • 目标: 让模型学会"思考 → 回答"的推理格式               │
│         │  • 训练: 3 epochs, 标准交叉熵 loss                     │
│         ▼                                                       │
│  Cold-Start Model (有了基础推理能力)                              │
│         │                                                       │
│         │ 阶段 2: GRPO (Group Relative Policy Optimization)     │
│         │  • 31K 条 (problem, answer) 数据                       │
│         │  • 每个 prompt 让模型自己生成 N=10 个回答               │
│         │  • Reward: 数学题答对=1, 答错=0                        │
│         │  • 目标: 强化"对的输出方式", 抑制"错的输出方式"           │
│         │  • 训练: 2 episodes, PPO 类算法                       │
│         ▼                                                       │
│  最终模型 (Qwen2.5VL-RLCS)                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 为什么需要"Cold Start"？

直接 RL（不做 SFT）的问题：
- 基础模型还没学会**结构化推理格式**（思考过程 + `\boxed{}` 答案）
- Reward 信号太稀疏：模型瞎答 99% 都错，几乎没法学
- 训练初期梯度噪声极大，容易崩

SFT 先让模型：
- 输出**带思考过程**的回答（"step 1: ... step 2: ... \boxed{answer}"）
- 输出**符合 reward 函数能识别**的格式（能从 `\boxed{}` 里提答案）
- 准确率不为零，让 GRPO 有学习信号

### 1.4 项目目录结构

```
RL-with-Cold-Start/
├── SFT/                              # 第一阶段：监督微调
│   ├── qwen2.5vl_sft.sh              # SFT 启动脚本（你要改的）
│   ├── convert_data.py               # 下载并转换 HF 数据集
│   ├── sys_prompt.txt                # 系统提示词
│   ├── data/Multimodal-Cold-Start.json   # 51K 条 SFT 数据
│   ├── output/                        # 训练输出 (checkpoint, logs)
│   └── swift/                         # ms-swift 框架源码（不用动）
│
├── GRPO/                             # 第二阶段：强化学习
│   ├── examples/
│   │   ├── qwen2_5_vl_7b_grpo.sh     # GRPO 启动脚本（你要改的）
│   │   ├── config.yaml                # 详细超参配置
│   │   └── score_function/math.py     # Reward 函数
│   ├── data/                          # 你下载的 RL 数据集
│   │   ├── Multimodal-RL-Data/       # 31K 条 RL 训练数据
│   │   └── geometry3k/                # 验证集
│   ├── checkpoints/                   # GRPO 训练输出
│   └── verl/                          # EasyR1/veRL 框架源码（不用动）
│
└── README.md
```

---

## 第二章 Qwen2.5-VL：模型如何"看图"

理解后面的训练之前，必须先搞清楚 Qwen2.5-VL 是怎么处理图像的，因为**显存、batch size、训练速度都跟这个直接相关**。

### 2.1 核心思想

> **把图像编码成一串向量，让这串向量和文本 token 的 embedding 在同一空间，然后插到文本序列里，让 LLM 当作"另一段输入"统一处理。**

LLM 本身只懂 token 序列，所以图像必须先"伪装"成 token。

### 2.2 完整流水线

```
原始图像 (任意分辨率)
     │
     ▼
[Step 1] Smart Resize  ← 关键：保留长宽比 + 对齐 28 的倍数
     │
     ▼
[Step 2] Patchify      ← 14×14 像素一个 patch
     │
     ▼
[Step 3] ViT 编码       ← Vision Transformer
     │
     ▼
[Step 4] 2×2 Spatial Merge  ← 4 个 patch 合成 1 个，压缩 4 倍
     │
     ▼
[Step 5] MLP Projector  ← 投影到 LLM hidden_dim
     │
     ▼
[Step 6] 替换文本中的 <image_pad> 占位符
     │
     ▼
统一进入 LLM Transformer，与文本 token 一起被处理
```

### 2.3 Step 1: Smart Resize

代码位置：`qwen_vl_utils/vision_process.py` 的 `smart_resize` 函数。

```python
def smart_resize(height, width, factor=28, min_pixels=3136, max_pixels=12845056):
    # 三个约束：
    # 1) 总像素 ∈ [min_pixels, max_pixels]
    # 2) 长宽都对齐到 28 的倍数
    # 3) 尽量保持原长宽比
    ...
```

**你脚本里 `MAX_PIXELS=1204224` 就是这里的 `max_pixels`。**

实际效果：

| 原图 | 缩放后 | 像素数 | 备注 |
|------|--------|-------|------|
| 800×600 | 784×588 | 460,992 | 没超 max，小改对齐到 28 倍数 |
| 1920×1080 | 1456×812 | 1,182,272 | 超了 max，等比缩小 |
| 50×40 | 84×56 | 4,704 | 低于 min=3136，放大 |

### 2.4 Step 2-3: Patchify + ViT 编码

```
1344×896 图像
  ↓ patchify (14×14)
96 × 64 = 6144 个 patch
  ↓ ViT (32 层 Transformer)
[6144, 1280] 向量
```

Qwen2.5-VL 的 ViT 有几个**关键创新**：

| 创新 | 作用 |
|------|-----|
| **Window Attention** | 大部分层只在 8×8 patch 窗口内 attention，省 80% 计算 |
| **2D RoPE** | 给 patch 的 (h, w) 坐标分别编码，模型知道空间关系 |
| **任意分辨率** | 不固定 224×224，能处理各种尺寸的图 |

### 2.5 Step 4-5: Spatial Merge + Projector

```
[6144, 1280] ViT 输出
    ↓ 2×2 spatial merge (压缩 4 倍)
[1536, 5120] 合并后
    ↓ MLP Projector (5120 → 2048)
[1536, 2048] 视觉 token，维度对齐到 LLM hidden_dim
```

**这一步对训练显存影响巨大**：
- `MAX_PIXELS=1204224` → 平均 ~1500 个视觉 token
- `MAX_PIXELS=602112` → 平均 ~750 个视觉 token
- 序列变短一半，attention 计算量降为 1/4

### 2.6 Step 6: 视觉 + 文本融合

```
文本：<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Find x...
                                       ↑ 这个占位符会被复制 1536 次

替换后：
  位置 0-2:   im_start, user, \n         ← 文本 embedding
  位置 3-1538: 视觉 token (1536 个)      ← ViT 输出
  位置 1539+: Find, x, ...              ← 文本 embedding
```

### 2.7 LLM 内部：图文统一处理

进了 LLM 之后，**模型完全不区分这是图还是文字**，统一在 Transformer 里做 attention。模型通过 attention 权重"看图"——比如生成"x = 5"时，attention 会聚焦在图中的相关边和数字上。

### 2.8 3D MRoPE（Multi-modal RoPE）

普通 LLM 用 1D 位置编码（"第几个 token"），Qwen2.5-VL 用 **3D**：

```
每个 token 的位置 = (t, h, w)

文本 "Hello":          (t=0, h=0, w=0)
图像 token (cell[5,3]): (t=1, h=5, w=3)
文本 "Find":            (t=L, h=0, w=0)
```

**好处**：模型能区分"这个视觉 token 在图的哪个位置"，并保持文本时序。

### 2.9 显存预估速查表

| MAX_PIXELS | 平均视觉 tokens | 典型总序列长 | 3B 训练显存 (per_device_bs=4) |
|-----------|----------------|------------|-----------------------------|
| 1204224 | ~1500 | ~2700 | ~55 GB/卡 |
| 602112 | ~750 | ~1900 | ~40 GB/卡 |
| 200704 | ~250 | ~1400 | ~28 GB/卡 |

> Attention 是 O(n²)，序列减半 → 计算量降 75%

---

## 第三章 SFT 阶段：让模型学会推理格式

### 3.1 一句话理解 SFT

> **SFT 让模型"模仿"一批高质量的问答对：通过对比模型输出和标准答案，调整模型权重让生成越来越接近标准答案。**

形式化：
```
loss = -log P(response | query)
     = -∑ log P(token_t | query, response_<t)
       t∈response
```

最小化"生成正确 response 中每个 token 的负对数概率"。

### 3.2 训练数据

**位置**：`SFT/data/Multimodal-Cold-Start.json` (51,534 条)

**来源**：原作者从 Geometry3K / GeoQA / ChartQA / TabMWP / ScienceQA / AI2D 等数据集采样问题，用 Qwen2.5-VL-32B 作 teacher 通过 **rejection sampling 蒸馏**：

```
图像 + 问题
   ↓
Teacher (Qwen2.5-VL-32B) 多次生成 CoT (最多 24 次)
   ↓
检查最终答案是否正确
   ↓
保留正确的轨迹 → 训练数据
```

**数据格式**：
```json
{
  "query": "<image>Find x. Round to the nearest tenth.\nChoices:\n8\n9\n10\n11",
  "response": "We are tasked with finding... \\boxed{11}",
  "images": ["data/Multimodal-Cold-Start/images/0.jpg"]
}
```

### 3.3 数据流：从 JSON 到 GPU 的完整路径

#### Step 1: 应用 chat template + system prompt

```
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Find x...<|im_end|>
<|im_start|>assistant
We are tasked with finding...
\boxed{11}<|im_end|>
```

#### Step 2: Tokenize

```python
input_ids = [151644, 8948, 198, 5501, ...]   # 假设 1200 个 token
```

#### Step 3: 图像处理（见第二章）

#### Step 4: 构造 `labels` —— **SFT 的核心 trick**

```python
input_ids = [token_1, ..., token_N]
labels    = [-100, ..., -100, response_token_1, ..., response_token_M, -100]
            └── system + user 部分 ──┘ └─ 只有 assistant 部分算 loss ─┘
```

`-100` 是 PyTorch 的"忽略 token"，CrossEntropyLoss 看到 -100 就跳过。

> **关键**：模型不需要学会预测用户的问题，只需要学会生成助手的回答。这是 instruction tuning 的本质。

### 3.4 前向 + 反向：模型内部发生了什么

```python
# 前向
outputs = model(input_ids, pixel_values, image_grid_thw, labels=labels)
loss = outputs.loss  # cross-entropy, 只在 labels != -100 的位置算

# 反向
loss.backward()         # 算梯度
optimizer.step()        # AdamW 更新参数
scheduler.step()        # 学习率衰减
optimizer.zero_grad()   # 清梯度
```

### 3.5 Batch、梯度累积、ZeRO-2 的协同

你的实际配置：
```bash
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
n_gpus = 4
effective_batch_size = 4 × 8 × 4 = 128
```

#### 一个 step 的执行流程

```
                  GPU 0           GPU 1           GPU 2           GPU 3
                  ─────           ─────           ─────           ─────
micro-step 1/8: forward 4 条 │ forward 4 条 │ forward 4 条 │ forward 4 条
                backward      │ backward      │ backward      │ backward
                累积梯度（不更新）
micro-step 2/8: forward 4 条 │ forward 4 条 │ forward 4 条 │ forward 4 条
                ...
micro-step 8/8: forward 4 条 │ forward 4 条 │ forward 4 条 │ forward 4 条
─────────────────────────────────────────────────────
                ALL-REDUCE 梯度（所有 GPU 同步）
                optimizer.step()  ← 一次参数更新
                ✅ global_step += 1
```

**"1 个 step" = 128 条数据 → 1 次参数更新**

#### ZeRO-2 在干什么

```
普通训练（每张卡都存全套）：
  GPU 0: 完整模型 + 完整梯度 + 完整 Adam(m,v)
  GPU 1: 完整模型 + 完整梯度 + 完整 Adam(m,v)
  ...

ZeRO-2（梯度切分 + 优化器状态切分）：
  GPU 0: 完整模型 + 梯度的 1/4 + Adam(m,v) 的 1/4
  GPU 1: 完整模型 + 梯度的 1/4 + Adam(m,v) 的 1/4
  ...
```

3B 模型 Adam 状态 ~24GB → ZeRO-2 切到 4 卡每卡只占 6GB，**省 18GB/卡**。

### 3.6 总 step 数计算

```
total_steps = ⌈ N_samples / effective_batch_size ⌉ × num_epochs
            = ⌈ 51534 / 128 ⌉ × 3
            = 403 × 3
            = 1209 steps
```

实际 1197（drop_last 或 packing 导致小偏差）。

### 3.7 SFT 监控指标（SwanLab）

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `train/loss` | 交叉熵 loss | 从 ~2.0 平滑下降到 ~0.5 |
| `train/grad_norm` | 梯度范数 | 1~10 正常，>50 异常 |
| `train/learning_rate` | 当前学习率 | 5e-5 → 0 (linear decay) |
| `epoch` | 当前 epoch 进度 | 0 → 3 |

### 3.8 SFT 的核心局限

SFT 只是"模仿"：
- ✅ 学会输出格式（思考过程 + `\boxed{}`）
- ✅ 见过类型题目
- ❌ **不知道什么是"好答案"什么是"坏答案"**：所有 token 同等模仿
- ❌ **没有探索机制**：只能学已见过的解法
- ❌ **错误传播**：teacher 数据有错，模型也会学错

→ 这就是为什么后面要做 GRPO。

---

## 第四章 GRPO 阶段：用强化学习强化推理能力

### 4.1 RL for LLM 的核心想法

> **让模型自己生成回答，给每个回答打分（reward），然后用 PPO 类算法调整模型，使其更倾向于生成高分回答、避免低分回答。**

跟 SFT 的根本区别：
- SFT: 数据是"老师给的标准答案"，模型模仿
- RL: 数据是"模型自己生成的", reward 告诉它生成的好坏

### 4.2 GRPO 的核心创新：放弃 Critic，用 Group 内对比

#### 传统 PPO 的痛点

PPO（Proximal Policy Optimization）需要训练一个 **critic 网络**估计 V(s)（状态价值），用它算 advantage：

```
advantage = reward - V(state)
```

问题：
- Critic 是个跟 actor 同等大小的模型 → **显存翻倍**
- Critic 训练困难，质量不稳定 → 直接影响 actor

#### GRPO 的方案

**对每个 prompt，让模型生成 N 个回答（一个 group），用 group 内的均值代替 critic**：

```python
# 对一个 prompt 生成 N=10 个回答 r_1, r_2, ..., r_10
rewards = [R(r_1), R(r_2), ..., R(r_10)]   # 比如 [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]

# Advantage = 标准化后的 reward
mean = rewards.mean()  # 0.6
std  = rewards.std()   # 0.5
advantages = (rewards - mean) / std
            # = [0.8, -1.2, 0.8, 0.8, -1.2, -1.2, 0.8, -1.2, 0.8, 0.8]
```

**含义**：在这 10 个回答里，**比平均水平好的回答 → 正 advantage（强化）**，**比平均水平差的 → 负 advantage（抑制）**。

#### 项目里的具体实现

代码位置：[GRPO/verl/trainer/core_algos.py:138-175](GRPO/verl/trainer/core_algos.py#L138-L175)

```python
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, eps=1e-6):
    scores = token_level_rewards.sum(dim=-1)   # 每个回答的总 reward
    id2score = defaultdict(list)

    # 把同一个 prompt 的所有回答归到一组
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # 每组算均值、方差
    for idx in id2score:
        id2mean[idx] = torch.mean(id2score[idx])
        id2std[idx] = torch.std(id2score[idx])

    # 标准化 = 个体 advantage
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns
```

### 4.3 Reward 函数（这个项目里只用准确率）

代码位置：[GRPO/examples/score_function/math.py](GRPO/examples/score_function/math.py)

```python
def compute_score(predict_str, ground_truth, format_weight=0.1):
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": accuracy_score,   # 训练时用 overall
        "format": 0,
        "accuracy": accuracy_score,
    }

def accuracy_reward(predict_str, ground_truth):
    answer = extract_boxed_content(predict_str)  # 从 \boxed{} 里提答案
    return 1.0 if grade_answer(answer, ground_truth) else 0.0
```

简单粗暴：**答对 → reward = 1，答错 → reward = 0**。

> 这种 binary outcome reward 配合 GRPO 效果好的原因：group 内对比会把"答对的回答"提升，"答错的回答"抑制，自动产生学习信号。

### 4.4 一个完整 GRPO step 的内部流程

以你的配置为例：
- `rollout_batch_size = 512`：每 step 用 512 个 prompts
- `rollout.n = 10`：每个 prompt 生成 10 个回答
- `global_batch_size = 128`：每次 PPO update 用 128 条

#### Phase 1: Rollout（vLLM 推理生成）

```
512 个 prompts
  ↓ 用 vLLM 高效推理（tensor_parallel_size=2）
512 × 10 = 5120 个回答
  ↓ 每个回答最长 2048 token
```

为什么用 vLLM 而不是直接用 actor 模型生成？**速度快 5-10 倍**（PagedAttention、continuous batching）。

#### Phase 2: 计算 reward

```python
for each (prompt, response) pair:
    score = compute_score(response, ground_truth)   # 0 或 1
```

#### Phase 3: 计算 reference logprob（用于 KL 散度）

```python
# Reference model = SFT 后的初始模型（冻结）
# 用 actor 当前的回答，问 reference: "你会用多大概率生成这些 token？"
ref_logprobs = reference_model.compute_log_prob(prompts, responses)
```

**为什么需要这个**？因为后面 PPO 的 loss 里有个 KL 约束项，**防止模型偏离 reference 太远**（避免训歪）。

#### Phase 4: 计算 old logprob（用于 PPO ratio）

```python
# Actor 当前权重下，重新算一遍 logprob（rollout 完后冻结这个值）
old_logprobs = actor.compute_log_prob(prompts, responses)
```

#### Phase 5: 计算 advantage（GRPO 核心）

见 4.2 的代码——对 5120 个回答按 prompt 分组，组内标准化。

#### Phase 6: PPO Update（多次 mini-batch）

```python
for epoch in range(ppo_epochs):
    for mini_batch in batches_of_128:
        # 当前 actor 权重重新算 logprob
        new_logprobs = actor.compute_log_prob(mini_batch)

        # PPO ratio
        ratio = exp(new_logprobs - old_logprobs)

        # PPO clipped objective
        loss_1 = ratio * advantage
        loss_2 = clip(ratio, 1-eps, 1+eps) * advantage
        policy_loss = -min(loss_1, loss_2).mean()

        # KL penalty (防止偏离 reference)
        kl_loss = kl_divergence(new_logprobs, ref_logprobs)

        # 总 loss
        loss = policy_loss + kl_coef * kl_loss
        loss.backward()
        optimizer.step()
```

**5120 个样本 / 128 = 40 次 mini-batch update**

### 4.5 GRPO 训练所需的"模型们"

GRPO 在显存里同时存在 **3 个 Qwen2.5-VL 副本**：

| 角色 | 作用 | 训练 | 显存 |
|------|-----|------|------|
| **Actor** | 当前正在训练的模型 | ✅ 训 | 占大头 |
| **Reference** | SFT 后冻结的模型 | ❌ 冻结 | 中等 |
| **vLLM Engine** | 生成回答用 | 与 Actor 共享权重 | KV cache 大 |

> 没有 Critic！这就是 GRPO 比 PPO 省一半显存的关键。

### 4.6 4 张 A100 上的资源分配

```
GPU 0 │ GPU 1 │ GPU 2 │ GPU 3
─────────────────────────────────
Actor (FSDP 全分片)
Reference (FSDP 全分片, CPU offload)
─────────────────────────────────
vLLM Engine 1 (TP=2) │ vLLM Engine 2 (TP=2)
GPU 0+1               │ GPU 2+3
```

- **FSDP** = Fully Sharded Data Parallel，类似 ZeRO-3 但 PyTorch 原生
- **CPU offload**：Reference 模型平时放 CPU 内存，用时才搬到 GPU
- **vLLM TP=2**：每个 vLLM 实例占 2 张卡做 tensor parallel

### 4.7 总 step 数计算

```
total_steps = ⌊ N_train / rollout_batch_size ⌋ × total_episodes
            = ⌊ 31818 / 512 ⌋ × 2
            = 62 × 2
            = 124 steps
```

`drop_last=True`，所以丢掉最后 73 个不满 512 的样本。

### 4.8 GRPO 监控指标（SwanLab）

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `critic/score/mean` | 当前 batch 的平均 reward（准确率）| 应该缓慢上升 |
| `actor/pg_loss` | PPO 策略 loss | 振荡正常，不应持续上升 |
| `actor/kl_loss` | 与 reference 的 KL 散度 | 0~0.1 健康，>1 偏离严重 |
| `actor/entropy` | 模型输出多样性 | 缓慢下降正常，骤降是塌陷 |
| `actor/grad_norm` | 梯度范数 | 0.1~5 正常 |
| `response_length/mean` | 生成长度均值 | 通常会先变长后稳定 |
| `val/test_score/mean` | 验证集准确率 | 整体趋势上升 |

### 4.9 GRPO 训练的典型曲线

```
reward (val accuracy)
   │
0.8│                              ╭───────
   │                          ╭───╯
0.6│                     ╭────╯
   │                ╭────╯
0.4│           ╭────╯
   │       ╭───╯
0.2│  ╭────╯
   │──╯
   └──────────────────────────────────────── step
   0    20    40    60    80    100   120
```

如果 reward 不涨甚至下降，常见原因：
- KL 系数太小 → 模型偏离 reference，能力崩塌
- 学习率太大
- Reward 信号不够（数据太难，全是 0 分）

---

## 第五章 实战手册：从环境配置到训练完成

### 5.1 环境配置

#### 5.1.1 创建 conda 环境

```bash
conda create -n coldstart python=3.10
conda activate coldstart
```

#### 5.1.2 安装核心依赖

```bash
# PyTorch (CUDA 12.4，对应大多数现代 A100 驱动)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# DeepSpeed (SFT 需要)
pip install deepspeed

# Qwen-VL 工具（注意版本！0.0.14+ 改了 API 与 ms-swift 不兼容）
pip install "qwen_vl_utils==0.0.8"
pip install qwen_omni_utils decord av librosa soundfile icecream

# SwanLab 实验跟踪
pip install swanlab

# 安装本地 ms-swift（SFT 框架）
cd SFT && pip install -e . && cd ..

# 安装本地 verl（GRPO 框架）
cd GRPO && pip install -e . && cd ..

# vLLM（GRPO 需要）
pip install "vllm>=0.5.1"
```

#### 5.1.3 验证环境

```bash
python3 -c "
import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
import deepspeed; print('deepspeed:', deepspeed.__version__)
import swanlab; print('swanlab:', swanlab.__version__)
from qwen_vl_utils import vision_process; print('qwen_vl_utils OK')
"
```

期望输出：
```
torch: 2.5.1+cu124 cuda: True
deepspeed: 0.19.1
swanlab: 0.8.2
qwen_vl_utils OK
```

### 5.2 SwanLab 登录

```bash
swanlab login   # 粘贴 API key (https://swanlab.cn 设置 → API Key)
```

凭据保存在 `~/.swanlab/.netrc`，只需登录一次。

### 5.3 SFT 阶段实战

#### 5.3.1 准备数据

```bash
cd SFT
python convert_data.py    # 下载 ~51K 条数据 + 图片 (约 5-10 分钟)
```

输出：
- `data/Multimodal-Cold-Start.json` (~95 MB)
- `data/Multimodal-Cold-Start/images/*.jpg` (51534 张图)

#### 5.3.2 配置启动脚本

[SFT/qwen2.5vl_sft.sh](SFT/qwen2.5vl_sft.sh)：

```bash
MAX_PIXELS=1204224 CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
  --model_type qwen2_5_vl \
  --model models/Qwen2.5-VL-3B-Instruct \
  --freeze_aligner false \
  --train_type full \
  --dataset data/Multimodal-Cold-Start.json \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --save_total_limit 2 \
  --eval_steps 200 \
  --save_steps 200 \
  --output_dir output \
  --system 'sys_prompt.txt' \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --deepspeed zero2 \
  --save_only_model false \
  --report_to tensorboard swanlab \
  --swanlab_project 'multimodal-cold-start' \
  --swanlab_exp_name 'qwen2.5vl-3b-sft' \
  --swanlab_mode cloud
```

关键参数解释：

| 参数 | 含义 |
|------|------|
| `MAX_PIXELS=1204224` | 图像最大像素（~1100×1100）|
| `--freeze_aligner false` | 视觉投影层也训练 |
| `--train_type full` | 全参数微调（vs lora）|
| `--per_device_train_batch_size 4` | 单卡 batch=4 |
| `--gradient_accumulation_steps 8` | 累积 8 次再更新（有效 bs=128）|
| `--save_only_model false` | 保留 optimizer 状态，支持断点续训 |
| `--deepspeed zero2` | DeepSpeed ZeRO-2（梯度+优化器切分）|

#### 5.3.3 启动训练

```bash
bash qwen2.5vl_sft.sh
```

观察项：
- 启动后会打印 `swanlab: 🚀 View run at https://...`，点击实时看曲线
- 另开终端 `nvidia-smi -l 2` 监控显存
- ~ 11-13 小时完成 3 个 epoch（1197 step）

#### 5.3.4 训练中断后续训

```bash
# 在脚本最后加一行
--resume_from_checkpoint output/v0-<时间戳>/checkpoint-XXX
```

#### 5.3.5 训练完成后

输出在 `SFT/output/v0-<时间戳>/checkpoint-1197/`，包含：
- `model.safetensors`（最终权重）
- `optimizer.pt`（Adam 状态，可用于续训）
- `args.json`（完整训练配置）

### 5.4 GRPO 阶段实战

#### 5.4.1 准备数据

```bash
cd GRPO
huggingface-cli download WaltonFuture/Multimodal-RL-Data \
    --repo-type dataset --local-dir data/Multimodal-RL-Data
huggingface-cli download hiyouga/geometry3k \
    --repo-type dataset --local-dir data/geometry3k
```

#### 5.4.2 配置 [config.yaml](GRPO/examples/config.yaml)

关键修改：把 logger 改成 swanlab，project 与 SFT 对齐：

```yaml
trainer:
  total_episodes: 15        # 实际由脚本覆盖
  logger: ["console","swanlab"]
  project_name: multimodal-cold-start
  experiment_name: qwen2_5_vl_7b_grpo
```

#### 5.4.3 配置启动脚本

[GRPO/examples/qwen2_5_vl_7b_grpo.sh](GRPO/examples/qwen2_5_vl_7b_grpo.sh)：

```bash
set -x

MODEL_PATH=../SFT/output/v0-<时间戳>/checkpoint-1197   # SFT 后的模型
EXP_NAME=qwen2_5_vl_3b_grpo
PROJECT_NAME=multimodal-cold-start
SAVE_PATH=./checkpoints/${PROJECT_NAME}/${EXP_NAME}

export SWANLAB_DIR=./swanlab_log

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data/Multimodal-RL-Data/data \
    data.val_files=./data/geometry3k/data/test-00000-of-00001.parquet \
    data.rollout_batch_size=512 \
    data.val_batch_size=500 \
    data.max_pixels=1204224 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.n=10 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.reward.score_function=./examples/score_function/math.py:compute_score \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=2 \
    trainer.save_freq=20 \
    trainer.save_limit=3 \
    trainer.val_freq=20 \
    trainer.val_before_train=true \
    trainer.save_checkpoint_path=${SAVE_PATH}
```

#### 5.4.4 启动训练

```bash
bash examples/qwen2_5_vl_7b_grpo.sh
```

观察项：
- 第一步会做 `val_before_train` 跑 baseline 验证（看 RL 起点）
- 124 step 总训练，约 11-13 小时
- SwanLab 上重点看 `critic/score/mean` 和 `val/test_score/mean`

#### 5.4.5 合并 checkpoint

verl 保存的是 sharded checkpoint，要合并成 HF 格式：

```bash
python3 scripts/model_merger.py \
    --local_dir checkpoints/multimodal-cold-start/qwen2_5_vl_3b_grpo/global_step_120/actor
```

输出标准 HuggingFace 格式模型，可直接用 transformers 加载。

### 5.5 实验追踪：SwanLab 使用

#### 5.5.1 在线查看

打开 https://swanlab.cn/@你的用户名/multimodal-cold-start

会看到：
- SFT run: `qwen2.5vl-3b-sft`
- GRPO run: `qwen2_5_vl_3b_grpo`

可以**并排对比两个阶段**的指标变化。

#### 5.5.2 导入历史 TensorBoard 日志

如果之前的训练没用 SwanLab，事后导入：

```bash
swanlab convert -t tensorboard \
  --tb-log-dir <训练目录>/runs \
  --project multimodal-cold-start
```

#### 5.5.3 离线模式

服务器无外网时：

```bash
# 启动时设置
--swanlab_mode local   # SFT
# 或
export SWANLAB_MODE=offline   # GRPO

# 之后有网时同步
swanlab sync <swanlog 目录>
```

---

## 第六章 排错与调优指南

### 6.1 SFT 阶段常见问题

#### 6.1.1 `ModuleNotFoundError: No module named 'qwen_vl_utils'`

```bash
pip install "qwen_vl_utils==0.0.8"
```

⚠️ **必须装 0.0.8，不能装最新版**。0.0.14+ 改了 API（去掉了 `IMAGE_FACTOR` 等常量），与 ms-swift 不兼容。

#### 6.1.2 `AttributeError: module 'qwen_vl_utils.vision_process' has no attribute 'IMAGE_FACTOR'`

同上，降级 qwen_vl_utils 到 0.0.8。

#### 6.1.3 `PackageNotFoundError: 'deepspeed' distribution was not found`

```bash
pip install deepspeed
```

#### 6.1.4 CUDA available: False

通常是 PyTorch 与系统驱动 CUDA 版本不匹配：

```bash
# 查驱动版本
nvidia-smi | head -3
# 看 CUDA Version: 12.X

# 装匹配版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

#### 6.1.5 OOM (Out of Memory)

按这个顺序调小：

```bash
# 1. 减小 per_device batch
--per_device_train_batch_size 2 --gradient_accumulation_steps 16

# 2. 降低图像分辨率
MAX_PIXELS=602112

# 3. 用更激进的 ZeRO
--deepspeed zero3   # 模型权重也切分（更省显存但更慢）

# 4. 开 gradient checkpointing (默认已开)
--gradient_checkpointing true
```

#### 6.1.6 训练 loss 不下降

- 学习率太小：默认 5e-5 对 3B 一般够，可试 1e-4
- 数据质量问题：检查 `args.json` 里 system prompt 和 dataset 是否对
- 数据没正确截断：检查 `--max_length`，太短会丢 response

### 6.2 GRPO 阶段常见问题

#### 6.2.1 vLLM 启动失败

```bash
# 查 vllm 版本
pip show vllm
# 与 torch 版本要兼容，推荐 vllm>=0.6.0 + torch 2.5.1
```

#### 6.2.2 Ray 启动失败

```bash
# 手动启 ray
ray start --head --port=6379
# 然后再跑训练
```

#### 6.2.3 显存不够（GRPO 特别吃显存）

按这个顺序调：

```bash
# 1. 降低 vLLM 显存占用
worker.rollout.gpu_memory_utilization=0.5

# 2. 减小 micro batch
worker.actor.micro_batch_size_per_device_for_update=2
worker.actor.micro_batch_size_per_device_for_experience=4

# 3. 开 reference offload
worker.ref.offload.offload_params=true   # 已默认开

# 4. 降低图像分辨率
data.max_pixels=602112

# 5. 减小 rollout batch
data.rollout_batch_size=256
worker.actor.global_batch_size=64   # 同步减小
```

#### 6.2.4 Reward 不涨

诊断流程：
1. 看 `val_before_train` 的初始 score —— 如果 < 10%，SFT 效果太差，先回去改 SFT
2. 看 `actor/kl_loss` —— 如果接近 0，模型没在学；如果暴涨，学崩了
3. 看 `actor/entropy` —— 如果骤降到 0，模型塌陷到单一输出
4. 看 `response_length/mean` —— 如果一直涨，可能在生成无意义长文骗 reward

调整建议：
- KL 太大：调小 `algorithm.kl_coef`（默认 1e-2）
- KL 太小：调大 `algorithm.kl_coef`
- 学崩：降低 `worker.actor.optim.lr`（默认 1e-6 已经很小）

### 6.3 性能调优

#### 6.3.1 加速 SFT

| 技巧 | 加速比 | 注意 |
|------|-------|------|
| 开 flash-attention | 1.5-2× | 装 `flash-attn` 包 |
| 数据 packing | 1.3-1.5× | 把短样本拼成长样本，少 padding |
| 降低 MAX_PIXELS | 1.5-3× | 牺牲一点视觉精度 |

#### 6.3.2 加速 GRPO

| 技巧 | 加速比 | 注意 |
|------|-------|------|
| vLLM tensor_parallel_size 调大 | 1.2-1.5× | 需要更多卡 |
| 增大 gpu_memory_utilization | 1.1-1.3× | 注意 OOM |
| rollout n 减小到 6-8 | 1.3-1.5× | 牺牲 group 内方差估计精度 |

### 6.4 实验设计建议

#### 6.4.1 第一次跑：先小规模 dry run

```bash
# SFT 缩减
--num_train_epochs 1 --save_steps 50 --max_steps 100

# GRPO 缩减
--trainer.max_steps=10 --data.rollout_batch_size=64
```

跑 1-2 小时确认流程通了再上完整训练。

#### 6.4.2 消融实验建议顺序

1. **Baseline**: 基础模型直接 GRPO（无 SFT）
2. **+ SFT only**: 只做 SFT，看模型推理能力
3. **+ SFT + GRPO** (论文方案): 看 GRPO 在 SFT 基础上的增益
4. **变化数据量**: 各阶段数据量减半，看是否还有效果

每次只改一个变量，SwanLab 上做并排对比。

---

## 附录 A：项目代码地图

### A.1 SFT 框架（ms-swift）

| 路径 | 作用 |
|------|------|
| [SFT/swift/cli/sft.py](SFT/swift/cli/sft.py) | SFT 命令入口 |
| [SFT/swift/llm/train/sft.py](SFT/swift/llm/train/sft.py) | SwiftSft 类（主训练逻辑）|
| [SFT/swift/llm/dataset/](SFT/swift/llm/dataset/) | 数据加载、tokenize、构造 labels |
| [SFT/swift/llm/model/model/qwen.py](SFT/swift/llm/model/model/qwen.py) | Qwen2.5-VL 模型加载逻辑 |
| [SFT/swift/llm/argument/](SFT/swift/llm/argument/) | 所有 CLI 参数定义 |
| [SFT/swift/trainers/](SFT/swift/trainers/) | Trainer 子类（基于 transformers.Trainer）|

### A.2 GRPO 框架（verl / EasyR1）

| 路径 | 作用 |
|------|------|
| [GRPO/verl/trainer/main.py](GRPO/verl/trainer/main.py) | GRPO 命令入口 |
| [GRPO/verl/trainer/ray_trainer.py](GRPO/verl/trainer/ray_trainer.py) | 主训练循环（Ray 协调多个 worker）|
| [GRPO/verl/trainer/core_algos.py](GRPO/verl/trainer/core_algos.py) | PPO/GRPO 算法核心（advantage 计算等）|
| [GRPO/verl/workers/](GRPO/verl/workers/) | Actor/Reference/Rollout worker 实现 |
| [GRPO/verl/utils/dataset.py](GRPO/verl/utils/dataset.py) | RL 数据加载 |
| [GRPO/verl/utils/logger/logger.py](GRPO/verl/utils/logger/logger.py) | SwanLab/wandb/tensorboard 后端 |
| [GRPO/examples/score_function/math.py](GRPO/examples/score_function/math.py) | Reward 函数 |
| [GRPO/examples/config.yaml](GRPO/examples/config.yaml) | 完整超参 YAML |

### A.3 关键文件用途速查

| 我要 ... | 改这个文件 |
|---------|-----------|
| 改 SFT 启动参数 | [SFT/qwen2.5vl_sft.sh](SFT/qwen2.5vl_sft.sh) |
| 改 SFT system prompt | [SFT/sys_prompt.txt](SFT/sys_prompt.txt) |
| 改 GRPO 启动参数 | [GRPO/examples/qwen2_5_vl_7b_grpo.sh](GRPO/examples/qwen2_5_vl_7b_grpo.sh) |
| 改 GRPO 默认超参 | [GRPO/examples/config.yaml](GRPO/examples/config.yaml) |
| 改 reward 计算 | [GRPO/examples/score_function/math.py](GRPO/examples/score_function/math.py) |
| 看训练实际配置 | `<output_dir>/args.json` |

---

## 附录 B：超参速查表

### B.1 SFT 关键超参

| 参数 | 默认/推荐值 | 含义 |
|------|------------|------|
| `--num_train_epochs` | 3 | 训练 epoch 数 |
| `--per_device_train_batch_size` | 4 (3B) / 2 (7B) | 单卡 batch |
| `--gradient_accumulation_steps` | 8-16 | 梯度累积步数 |
| `--learning_rate` | 5e-5 | 学习率 |
| `MAX_PIXELS` | 1204224 | 图像最大像素 |
| `--deepspeed` | zero2 | DeepSpeed ZeRO 级别 |
| `--save_steps` | 200 | 保存频率 |
| `--save_only_model` | false | 是否只存模型权重（false=可续训）|

### B.2 GRPO 关键超参

| 参数 | 默认/推荐值 | 含义 |
|------|------------|------|
| `data.rollout_batch_size` | 512 | 每 step 用多少 prompt 做 rollout |
| `data.max_pixels` | 1204224 | 图像最大像素 |
| `worker.actor.global_batch_size` | 128 | PPO update batch |
| `worker.actor.micro_batch_size_per_device_for_update` | 4 | 单卡 update micro bs |
| `worker.actor.optim.lr` | 1e-6 | actor 学习率（比 SFT 小 50x）|
| `worker.rollout.n` | 10 | GRPO group size（每 prompt 生成几个回答）|
| `worker.rollout.tensor_parallel_size` | 2 | vLLM 的 TP 大小 |
| `worker.rollout.gpu_memory_utilization` | 0.6 | vLLM KV cache 占显存比例 |
| `algorithm.kl_coef` | 1e-2 | KL 约束系数 |
| `trainer.total_episodes` | 2 | 总 episode 数 |
| `trainer.save_freq` | 20 | 保存频率（step）|
| `trainer.val_freq` | 20 | 验证频率（step）|

### B.3 计算公式速查

**SFT step 数**：
```
total_steps = ⌈ N_samples / (per_device_bs × n_gpus × grad_accum) ⌉ × num_epochs
```

**GRPO step 数**：
```
total_steps = ⌊ N_samples / rollout_batch_size ⌋ × total_episodes
```

**GRPO 每 step 生成数**：
```
generations_per_step = rollout_batch_size × rollout.n
```

**GRPO 每 step PPO update 数**：
```
updates_per_step = generations_per_step / global_batch_size
```

---

## 🎓 结语

这套两阶段方法（SFT cold start + RL）是当前训练推理型大模型的主流范式：
- **DeepSeek-R1**: SFT(部分) + RL
- **OpenAI o1**: 推测类似流程
- **Qwen-QwQ**: 类似思路

理解了这个流程，你不仅能复现本项目，**也能把这套方法迁移到任何"需要复杂推理"的领域**：代码生成、定理证明、医学诊断、智能体决策等。

### 下一步学习建议

1. **实验复现**：先按本教程跑通完整流程，对比 SwanLab 上 SFT 和 GRPO 两阶段的曲线
2. **消融实验**：尝试只 SFT 不 RL、只 RL 不 SFT，看效果差异
3. **自定义 Reward**：改 [math.py](GRPO/examples/score_function/math.py)，加入对推理过程的奖励（不止看最终答案）
4. **扩展数据**：用其他多模态推理数据集（如 MathVision、CharXiv）替换或加入训练
5. **深入算法**：读 [core_algos.py](GRPO/verl/trainer/core_algos.py) 理解 PPO/GRPO 实现，对比 REINFORCE/RLOO 等其他算法

### 论文与参考资料

- **本项目论文**：[arXiv:2505.22334](https://arxiv.org/pdf/2505.22334)
- **GRPO 原始论文**：[DeepSeekMath](https://arxiv.org/abs/2402.03300)
- **Qwen2.5-VL 技术报告**：[arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- **ms-swift 文档**：https://github.com/modelscope/ms-swift
- **EasyR1/verl 文档**：https://github.com/hiyouga/EasyR1

---

**Happy Training! 🚀**
