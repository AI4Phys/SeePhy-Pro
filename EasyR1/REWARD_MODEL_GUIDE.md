# Reward Model 部署指南

本文档详细说明如何在 EasyR1 框架中添加和部署自己的 Reward Model（奖励模型）。

## 目录
1. [架构概览](#架构概览)
2. [Actor/Critic 模型加载机制](#actorcritic-模型加载机制)
3. [GPU 资源分配](#gpu-资源分配)
4. [Reward Model 实现步骤](#reward-model-实现步骤)
5. [配置示例](#配置示例)
6. [完整示例代码](#完整示例代码)

---

## 架构概览

EasyR1 框架支持两种 Reward 计算方式：

1. **基于函数的 Reward**（当前默认）：
   - 使用 `AutoRewardManager` 加载 Python 函数
   - 在 CPU 上运行（通过 `num_cpus` 配置）
   - 适用于规则基础的 reward（如数学题答案检查）

2. **基于神经网络的 Reward Model**（需要实现）：
   - 使用 `FSDPWorker` 加载神经网络模型
   - 在 GPU 上运行，支持 FSDP 分布式训练
   - 适用于需要模型推理的 reward

---

## Actor/Critic 模型加载机制

### 1. FSDPWorker 架构

`FSDPWorker` 是核心的 worker 类，负责加载和管理模型。关键方法：

```python
# verl/workers/fsdp_workers.py

class FSDPWorker(Worker):
    def __init__(self, config, role):
        # role 可以是: "actor", "critic", "rollout", "ref", "reward"
        # 根据 role 决定加载哪些模型
        
    def _build_model_optimizer(self, model_config, fsdp_config, optim_config, padding_free, role):
        # 加载模型的核心方法
        # 1. 加载 tokenizer 和 processor
        # 2. 从 HuggingFace 加载模型
        # 3. 使用 FSDP 包装模型
        # 4. 创建优化器（如果需要）
        
    def init_model(self):
        # 根据 role 调用 _build_model_optimizer
        # 初始化对应的模型（actor/critic/ref/reward）
```

### 2. 模型加载流程

```python
# 1. 加载配置
model_config = AutoConfig.from_pretrained(model_path, ...)

# 2. 选择模型类
if role == "critic":
    AutoClass = AutoModelForTokenClassification
elif is_vision_model:
    AutoClass = AutoModelForImageTextToText
else:
    AutoClass = AutoModelForCausalLM

# 3. 加载模型
model = AutoClass.from_pretrained(
    model_path,
    config=model_config,
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2",
    device_map="cpu" if enable_rank0_init else "cuda",
    ...
)

# 4. FSDP 包装
fsdp_module = FSDP(
    model,
    sharding_strategy=sharding_strategy,
    mixed_precision=mixed_precision,
    device_mesh=device_mesh,
    ...
)
```

### 3. GPU 资源分配

通过 `ResourcePoolManager` 管理 GPU 资源：

```python
# verl/trainer/main.py

resource_pool_spec = {
    "global_pool": [n_gpus_per_node] * nnodes,  # 每个节点的 GPU 数量
}

mapping = {
    Role.ActorRolloutRef: "global_pool",
    Role.Critic: "global_pool",
    Role.RewardModel: "global_pool",  # 添加 reward model 映射
}
```

---

## GPU 资源分配

### 1. 资源池配置

```python
# 在 verl/trainer/main.py 中配置

resource_pool_spec = {
    "global_pool": [8] * 4,  # 4 个节点，每个节点 8 个 GPU
    # 或者为 reward model 单独分配资源池
    "reward_pool": [2] * 2,  # 2 个节点，每个节点 2 个 GPU
}

mapping = {
    Role.ActorRolloutRef: "global_pool",
    Role.Critic: "global_pool",
    Role.RewardModel: "reward_pool",  # 使用独立的资源池
}
```

### 2. FSDP 配置

在配置文件中设置 FSDP 参数：

```yaml
worker:
  reward_model:  # 新增 reward model 配置
    model:
      model_path: path/to/your/reward/model
      enable_gradient_checkpointing: false  # reward model 通常不需要梯度
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
      enable_rank0_init: true
    offload:
      offload_params: false  # reward model 推理不需要 offload
```

---

## Reward Model 实现步骤

### 步骤 1: 扩展 FSDPWorker 支持 Reward Model

在 `verl/workers/fsdp_workers.py` 中添加 reward model 支持：

```python
class FSDPWorker(Worker):
    def __init__(self, config, role):
        # ... 现有代码 ...
        
        self._has_reward_model = self.role == "reward"
        
        if self._has_reward_model:
            self._use_param_offload = self.config.reward_model.offload.offload_params
            self._init_dist_mesh(self.config.reward_model, "reward")
    
    def init_model(self):
        # ... 现有代码 ...
        
        if self._has_reward_model:
            self._build_model_optimizer(
                model_config=self.config.reward_model.model,
                fsdp_config=self.config.reward_model.fsdp,
                optim_config=None,  # reward model 不需要优化器
                padding_free=self.config.reward_model.padding_free,
                role="reward",
            )
            # reward model 不需要梯度
            self.fsdp_module.requires_grad_(False)
            self.fsdp_module.eval()
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reward_scores(self, data: DataProto):
        """计算 reward scores"""
        assert self._has_reward_model
        
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            
            # 前向传播计算 reward
            # 假设 reward model 输出 shape: (batch_size, seq_len)
            with torch.no_grad():
                outputs = self.fsdp_module(**data.batch)
                # 根据你的 reward model 结构调整
                reward_scores = outputs.logits  # 或 outputs.reward_scores
            
            output = DataProto.from_dict(tensors={"reward_scores": reward_scores})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        
        output = output.to("cpu")
        return output
```

### 步骤 2: 创建 Reward Model Worker 类

创建 `verl/workers/reward/model.py`：

```python
# verl/workers/reward/model.py

from typing import Tuple
import torch
from ...protocol import DataProto
from ...single_controller.base.decorator import register, Dispatch
from ..fsdp_workers import FSDPWorker


class RewardModelWorker(FSDPWorker):
    """Reward Model Worker，继承自 FSDPWorker"""
    
    def __init__(self, config, role="reward"):
        super().__init__(config, role)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict]:
        """
        计算 reward scores
        
        Returns:
            reward_tensor: (batch_size, seq_len) 的 reward scores
            reward_metrics: 包含额外指标的字典
        """
        reward_output = self.compute_reward_scores(data)
        reward_scores = reward_output.batch["reward_scores"]
        
        # 转换为 token-level rewards
        # 假设 reward_scores shape: (batch_size, seq_len)
        reward_tensor = reward_scores.float()
        
        # 提取额外的 metrics（如果有）
        reward_metrics = {}
        if "reward_metrics" in reward_output.batch:
            reward_metrics = reward_output.batch["reward_metrics"]
        
        return reward_tensor, reward_metrics
```

### 步骤 3: 更新 Trainer 以支持 Reward Model

在 `verl/trainer/ray_trainer.py` 中：

```python
class RayPPOTrainer:
    def __init__(self, ...):
        # ... 现有代码 ...
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        # 如果使用 reward model，则不需要 reward_fn
        if self.use_reward_model:
            self.reward_fn = None
    
    def init_workers(self):
        # ... 现有代码 ...
        
        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()
    
    def _make_batch_data(self, metrics):
        # ... 现有代码 ...
        
        # 计算 reward
        if self.use_reward_model:
            # 使用 reward model
            reward_output = self.rm_wg.compute_reward(new_batch)
            reward_tensor, reward_metrics = ray.get(reward_output)
        else:
            # 使用 reward function
            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
```

### 步骤 4: 更新配置类

在 `verl/workers/config.py` 中添加 reward model 配置：

```python
@dataclass
class RewardModelConfig:
    """Reward Model 配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    padding_free: bool = True
    dynamic_batching: bool = True
    ulysses_size: int = 1
    offload: OffloadConfig = field(default_factory=OffloadConfig)


@dataclass
class WorkerConfig:
    # ... 现有字段 ...
    reward_model: Optional[RewardModelConfig] = None
```

### 步骤 5: 更新 main.py

在 `verl/trainer/main.py` 中：

```python
def run(self, config: PPOConfig):
    # ... 现有代码 ...
    
    role_worker_mapping = {
        Role.ActorRolloutRef: ray.remote(FSDPWorker),
        Role.Critic: ray.remote(FSDPWorker),
    }
    
    # 如果配置了 reward model，添加到 mapping
    if config.worker.reward_model is not None:
        from ..workers.reward.model import RewardModelWorker
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
    
    # 更新 resource pool mapping
    mapping = {
        Role.ActorRolloutRef: global_pool_id,
        Role.Critic: global_pool_id,
    }
    
    if config.worker.reward_model is not None:
        # 可以选择共享资源池或独立资源池
        mapping[Role.RewardModel] = global_pool_id  # 或 "reward_pool"
    
    # 只有在没有 reward model 时才创建 reward function
    if config.worker.reward_model is None:
        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
    else:
        reward_fn = None
```

---

## 配置示例

### 配置文件示例

```yaml
# configs/config_with_reward_model.yaml

worker:
  actor:
    # ... actor 配置 ...
  
  reward_model:  # 新增 reward model 配置
    model:
      model_path: path/to/your/reward/model
      enable_gradient_checkpointing: false
      trust_remote_code: true
    fsdp:
      enable_full_shard: true
      enable_cpu_offload: false
      enable_rank0_init: true
      fsdp_size: 4  # reward model 的 FSDP 并行大小
    padding_free: true
    dynamic_batching: true
    ulysses_size: 1
    offload:
      offload_params: false
      offload_optimizer: false

  reward:  # 如果使用 reward function，保留此配置
    reward_function: null  # 设置为 null 表示不使用 reward function

trainer:
  # ... trainer 配置 ...
```

### 训练脚本示例

```bash
#!/bin/bash

python3 -m verl.trainer.main \
    config="configs/config_with_reward_model.yaml" \
    worker.reward_model.model.model_path="path/to/reward/model" \
    worker.reward_model.fsdp.fsdp_size=4 \
    # ... 其他参数 ...
```

---

## 完整示例代码

### 1. Reward Model Worker 完整实现

```python
# verl/workers/reward/model_worker.py

from typing import Tuple, Dict, Any
import torch
from collections import defaultdict

from ...protocol import DataProto
from ...single_controller.base.decorator import register, Dispatch
from ..fsdp_workers import FSDPWorker


class RewardModelWorker(FSDPWorker):
    """
    Reward Model Worker
    
    继承自 FSDPWorker，支持：
    - FSDP 分布式推理
    - 参数 offload
    - 多模态输入处理
    """
    
    def __init__(self, config, role="reward"):
        super().__init__(config, role)
        self._has_reward_model = True
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        计算 reward scores
        
        Args:
            data: 包含 prompts 和 responses 的 DataProto
            
        Returns:
            reward_tensor: (batch_size, seq_len) 的 token-level reward scores
            reward_metrics: 包含额外指标的字典
        """
        # 处理多模态输入
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        
        # 加载模型（如果使用了 offload）
        if self._use_param_offload:
            from ...utils.fsdp_utils import load_fsdp_model
            load_fsdp_model(self.fsdp_module)
        
        # 准备输入
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            
            # 构建模型输入
            model_inputs = {
                "input_ids": data.batch["input_ids"],
                "attention_mask": data.batch["attention_mask"],
            }
            
            # 添加多模态输入（如果有）
            if "multi_modal_inputs" in data.non_tensor_batch:
                # 处理图像/视频输入
                for i, mm_input in enumerate(data.non_tensor_batch["multi_modal_inputs"]):
                    if mm_input:
                        # 将多模态输入移动到 GPU
                        for k, v in mm_input.items():
                            if isinstance(v, torch.Tensor):
                                mm_input[k] = v.to(torch.cuda.current_device())
            
            # 前向传播
            with torch.no_grad():
                outputs = self.fsdp_module(**model_inputs)
                
                # 提取 reward scores
                # 根据你的 reward model 结构调整这里
                if hasattr(outputs, "reward_scores"):
                    reward_scores = outputs.reward_scores
                elif hasattr(outputs, "logits"):
                    # 如果输出是 logits，可能需要处理
                    reward_scores = outputs.logits.squeeze(-1)  # (batch_size, seq_len)
                else:
                    raise ValueError("Reward model output must have 'reward_scores' or 'logits'")
            
            # 创建输出 DataProto
            output = DataProto.from_dict(tensors={"reward_scores": reward_scores})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        
        # Offload 模型（如果使用了 offload）
        if self._use_param_offload:
            from ...utils.fsdp_utils import offload_fsdp_model
            offload_fsdp_model(self.fsdp_module)
        
        # 转换为 token-level rewards
        reward_tensor = output.batch["reward_scores"].float()
        
        # 构建 metrics
        reward_metrics = defaultdict(list)
        batch_size = reward_tensor.shape[0]
        
        # 计算每个样本的平均 reward
        response_mask = data.batch.get("response_mask", None)
        if response_mask is not None:
            # 只对 response tokens 计算平均 reward
            masked_rewards = reward_tensor * response_mask
            seq_lengths = response_mask.sum(dim=-1)
            avg_rewards = masked_rewards.sum(dim=-1) / seq_lengths.clamp(min=1)
            
            for i in range(batch_size):
                reward_metrics["overall"].append(avg_rewards[i].item())
        else:
            # 如果没有 mask，使用最后一个 token 的 reward
            for i in range(batch_size):
                reward_metrics["overall"].append(reward_tensor[i, -1].item())
        
        output = output.to("cpu")
        return reward_tensor, reward_metrics
```

### 2. 更新 FSDPWorker 以支持 Reward Model

在 `verl/workers/fsdp_workers.py` 的 `__init__` 方法中添加：

```python
def __init__(self, config, role):
    # ... 现有代码 ...
    
    self._has_reward_model = self.role == "reward"
    
    if self._has_reward_model:
        self._use_param_offload = self.config.reward_model.offload.offload_params
        self._init_dist_mesh(self.config.reward_model, "reward")
```

在 `init_model` 方法中添加：

```python
def init_model(self):
    # ... 现有代码 ...
    
    if self._has_reward_model:
        self._build_model_optimizer(
            model_config=self.config.reward_model.model,
            fsdp_config=self.config.reward_model.fsdp,
            optim_config=None,  # reward model 不需要优化器
            padding_free=self.config.reward_model.padding_free,
            role="reward",
        )
        # reward model 设置为评估模式，不需要梯度
        self.fsdp_module.requires_grad_(False)
        self.fsdp_module.eval()
```

---

## 关键要点总结

1. **模型加载**：Reward Model 使用与 Actor/Critic 相同的 `_build_model_optimizer` 方法加载
2. **GPU 分配**：通过 `ResourcePoolManager` 管理，可以共享或独立资源池
3. **FSDP 支持**：Reward Model 支持 FSDP 分布式推理，可以设置 `fsdp_size`
4. **参数 Offload**：支持参数 offload 以节省 GPU 内存
5. **多模态支持**：自动处理图像/视频等多模态输入
6. **推理模式**：Reward Model 始终在 `eval()` 模式下运行，不需要梯度

---

## 调试建议

1. **检查模型加载**：确保 reward model 路径正确，模型格式兼容
2. **检查 GPU 资源**：确保有足够的 GPU 分配给 reward model
3. **检查输出格式**：确保 reward model 输出格式符合预期（token-level scores）
4. **内存管理**：如果 GPU 内存不足，启用参数 offload
5. **分布式调试**：检查 FSDP 配置是否正确，确保所有 rank 都能访问模型

---

## 常见问题

**Q: Reward Model 和 Reward Function 可以同时使用吗？**
A: 不可以，两者是互斥的。如果配置了 `reward_model`，则 `reward` 配置会被忽略。

**Q: Reward Model 需要训练吗？**
A: 不需要。Reward Model 在推理模式下运行，不需要优化器。

**Q: 如何为 Reward Model 分配独立的 GPU？**
A: 在 `resource_pool_spec` 中创建独立的资源池，并在 `mapping` 中指定。

**Q: Reward Model 支持多模态输入吗？**
A: 支持。框架会自动处理图像/视频输入，与 Actor 模型相同。

---

## 参考文件

- `verl/workers/fsdp_workers.py`: FSDP Worker 实现
- `verl/workers/reward/function.py`: Reward Function 实现
- `verl/trainer/ray_trainer.py`: Trainer 实现
- `verl/trainer/main.py`: 主入口
- `configs/config.yaml`: 配置文件示例

