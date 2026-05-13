# Reward Model 使用指南

本文档说明如何使用已实现的基于神经网络的 Reward Model。

## 实现概览

已完成的实现包括：

1. **RewardModelConfig** (`verl/workers/reward/model_config.py`): Reward Model 配置类
2. **RewardModelWorker** (`verl/workers/reward/model_worker.py`): Reward Model Worker 实现
3. **FSDPWorker 扩展**: 支持 reward model role
4. **Trainer 更新**: 支持 Reward Model 调用
5. **Main 更新**: 自动检测并使用 Reward Model

## 快速开始

### 1. 配置文件设置

在配置文件中添加 `reward_model` 配置：

```yaml
worker:
  # Reward Model (神经网络)
  reward_model:
    model:
      model_path: path/to/your/reward/model
      enable_gradient_checkpointing: false
      trust_remote_code: true
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
      enable_rank0_init: true
      fsdp_size: 4  # FSDP 并行大小
    padding_free: true
    dynamic_batching: true
    ulysses_size: 1
    offload:
      offload_params: false  # 如果 GPU 内存不足，设为 true

  # Reward Function (Python 函数) - 如果设置了 reward_model，此配置会被忽略
  reward:
    reward_function: null
```

### 2. Reward Model 要求

你的 Reward Model 需要满足以下要求：

1. **输出格式**：模型输出必须包含以下之一：
   - `reward_scores`: 直接输出 reward scores，shape: `(batch_size, seq_len)`
   - `logits`: 输出 logits，shape: `(batch_size, seq_len, vocab_size)` 或 `(batch_size, seq_len, 1)`
   - 如果输出 `(batch_size, seq_len, vocab_size)`，会自动进行 mean pooling

2. **输入格式**：模型应该接受标准的 HuggingFace 输入：
   - `input_ids`: `(batch_size, seq_len)`
   - `attention_mask`: `(batch_size, seq_len)`
   - 如果支持多模态，还需要处理 `pixel_values` 等

3. **模型类型**：支持以下模型类型：
   - `AutoModelForCausalLM` (默认)
   - `AutoModelForImageTextToText` (如果检测到是视觉语言模型)

### 3. 运行训练

```bash
python3 -m verl.trainer.main \
    config="configs/config_reward_model_example.yaml" \
    worker.reward_model.model.model_path="path/to/your/reward/model" \
    # ... 其他参数
```

## 详细说明

### Reward Model vs Reward Function

框架支持两种 Reward 计算方式，它们是互斥的：

| 特性 | Reward Function | Reward Model |
|------|----------------|--------------|
| 实现方式 | Python 函数 | 神经网络模型 |
| 运行位置 | CPU | GPU |
| 分布式支持 | 否 | 是 (FSDP) |
| 配置字段 | `worker.reward` | `worker.reward_model` |
| 适用场景 | 规则基础 reward | 需要模型推理的 reward |

### 自动检测逻辑

框架会自动检测使用哪种方式：

1. 如果 `config.worker.reward_model` 不为 `None`，使用 Reward Model
2. 否则，使用 Reward Function

### GPU 资源分配

Reward Model 默认与 Actor/Critic 共享 GPU 资源池。如果需要独立分配：

```python
# 在 verl/trainer/main.py 中修改
resource_pool_spec = {
    "global_pool": [8] * 4,  # Actor/Critic 使用
    "reward_pool": [2] * 2,  # Reward Model 使用
}

mapping = {
    Role.ActorRolloutRef: "global_pool",
    Role.Critic: "global_pool",
    Role.RewardModel: "reward_pool",  # 独立资源池
}
```

### 自定义 Reward Model 输出处理

如果你的 Reward Model 输出格式不同，可以修改 `verl/workers/reward/model_worker.py` 中的 `compute_reward` 方法：

```python
# 在 compute_reward 方法中修改这部分
with torch.no_grad():
    outputs = self.fsdp_module(**model_inputs)
    
    # 根据你的模型输出格式调整
    if hasattr(outputs, "reward_scores"):
        reward_scores = outputs.reward_scores
    elif hasattr(outputs, "logits"):
        # 自定义处理逻辑
        reward_scores = your_custom_processing(outputs.logits)
    # ...
```

## 示例：创建一个简单的 Reward Model

### 1. 模型定义

```python
# reward_model.py
from transformers import AutoModelForCausalLM, AutoConfig
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        # 添加 reward head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        # 使用最后一个 hidden state
        last_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        # 通过 reward head
        reward_scores = self.reward_head(last_hidden).squeeze(-1)  # (batch_size, seq_len)
        return type('RewardOutput', (), {'reward_scores': reward_scores})()
```

### 2. 保存模型

```python
# 训练并保存你的 reward model
model = RewardModel("Qwen/Qwen2.5-7B-Instruct")
# ... 训练代码 ...
model.save_pretrained("path/to/your/reward/model")
```

### 3. 使用模型

在配置文件中设置：

```yaml
worker:
  reward_model:
    model:
      model_path: path/to/your/reward/model
      trust_remote_code: true  # 如果模型需要自定义代码
```

## 调试建议

1. **检查模型加载**：
   - 确保模型路径正确
   - 检查模型格式是否兼容 HuggingFace

2. **检查输出格式**：
   - 打印模型输出，确认有 `reward_scores` 或 `logits`
   - 检查输出 shape 是否符合预期

3. **GPU 内存**：
   - 如果内存不足，启用 `offload_params: true`
   - 调整 `fsdp_size` 以使用更多 GPU

4. **分布式调试**：
   - 检查 FSDP 配置是否正确
   - 确保所有 rank 都能访问模型

## 常见问题

**Q: Reward Model 和 Reward Function 可以同时使用吗？**
A: 不可以，它们是互斥的。如果设置了 `reward_model`，`reward` 配置会被忽略。

**Q: Reward Model 需要训练吗？**
A: 不需要。Reward Model 在推理模式下运行，不需要优化器。

**Q: 如何为 Reward Model 分配独立的 GPU？**
A: 修改 `resource_pool_spec` 和 `mapping`，为 Reward Model 创建独立的资源池。

**Q: Reward Model 支持多模态输入吗？**
A: 支持。框架会自动处理图像/视频输入，与 Actor 模型相同。

**Q: 如何自定义 Reward Model 的输出处理？**
A: 修改 `verl/workers/reward/model_worker.py` 中的 `compute_reward` 方法。

## 参考文件

- `verl/workers/reward/model_config.py`: Reward Model 配置
- `verl/workers/reward/model_worker.py`: Reward Model Worker 实现
- `verl/workers/fsdp_workers.py`: FSDP Worker (包含 reward model 支持)
- `verl/trainer/ray_trainer.py`: Trainer (包含 reward model 调用)
- `verl/trainer/main.py`: 主入口 (包含 reward model 注册)
- `configs/config_reward_model_example.yaml`: 配置示例

