# N-Gram 语言模型

一个结构化、可扩展的 N-Gram 语言模型实现，支持多种平滑技术和采样策略。

## 模块结构

```
ngram/
├── config.py        # 配置管理（模型参数、路径、随机种子等）
├── utils.py         # 工具函数（语料生成、数据集划分、日志配置、异常类）
├── ngram_model.py   # 核心模型类（NGramModel 及平滑方法）
├── demo.py          # 演示/主逻辑
├── test_ngram.py    # 单元测试
└── README.md        # 文档说明
```

## 快速开始

### 1. 训练模型

```python
from ngram_model import NGramModel
from utils import generate_corpus, split_corpus

# 生成语料库
corpus = generate_corpus()

# 划分训练集和测试集
train_corpus, test_corpus = split_corpus(corpus, train_ratio=0.8, random_seed=42)

# 创建并训练模型
model = NGramModel(n=2)
model.train(train_corpus)
```

### 2. 预测下一个词

```python
# 预测下一个概率最大的词
context = ["今", "天"]
next_word, probability = model.predict_next(context)
print(f"上下文 '{''.join(context)}' -> 预测: '{next_word}' (概率: {probability:.6f})")
```

### 3. 生成文本

```python
# 使用加权随机采样生成文本
text = model.generate_text(seed=["今", "天"], max_length=15)
print(f"生成的文本: {text}")

# 使用贪婪采样生成文本
from config import SamplingStrategy
text = model.generate_text(
    seed=["今", "天"], 
    max_length=15,
    sampling_strategy=SamplingStrategy.GREEDY
)
print(f"生成的文本: {text}")
```

### 4. 计算困惑度

```python
# 计算测试集上的困惑度
perplexity = model.calculate_perplexity(test_corpus)
print(f"测试集困惑度: {perplexity:.4f}")
```

## 配置参数说明

### ModelConfig（模型配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| n | int | 2 | N-Gram 的阶数 |
| smoothing_method | SmoothingMethod | ADD_ONE | 平滑方法 |
| smoothing_k | float | 1.0 | Add-k 平滑的 k 值 |

### TrainingConfig（训练配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| random_seed | int | 42 | 随机种子 |
| train_ratio | float | 0.8 | 训练集比例 |

### GenerationConfig（生成配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_length | int | 20 | 生成文本的最大长度 |
| sampling_strategy | SamplingStrategy | WEIGHTED_RANDOM | 采样策略 |

### LogConfig（日志配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| log_level | str | "INFO" | 日志级别 |
| log_to_file | bool | False | 是否输出到文件 |
| log_file_path | str | None | 日志文件路径 |

## 扩展平滑方式

### 使用内置平滑方法

```python
from config import ModelConfig, SmoothingMethod
from ngram_model import NGramModel

# Add-1 (Laplace) 平滑
config = ModelConfig(n=2, smoothing_method=SmoothingMethod.ADD_ONE)
model = NGramModel(config=config)

# Add-k 平滑
config = ModelConfig(n=2, smoothing_method=SmoothingMethod.ADD_K, smoothing_k=0.5)
model = NGramModel(config=config)

# 无平滑（最大似然估计）
config = ModelConfig(n=2, smoothing_method=SmoothingMethod.NONE)
model = NGramModel(config=config)
```

### 自定义平滑方法

```python
from ngram_model import BaseSmoothing
from typing import Dict, Tuple
from collections import Counter

class MySmoothing(BaseSmoothing):
    """自定义平滑方法"""
    
    def calculate_probability(
        self, 
        word: str, 
        context: Tuple[str, ...],
        ngram_counts: Dict[Tuple[str, ...], Counter],
        context_counts: Counter,
        vocab_size: int
    ) -> float:
        # 实现你的平滑逻辑
        count_context_word = ngram_counts[context].get(word, 0)
        count_context = context_counts.get(context, 0)
        # 示例：简单的 Good-Turing 平滑
        return (count_context_word + 0.5) / (count_context + vocab_size * 0.5)
```

## 扩展采样策略

### 使用内置采样策略

```python
from config import SamplingStrategy

# 贪婪采样：每次选择概率最高的词
text = model.generate_text(
    seed=["今", "天"],
    sampling_strategy=SamplingStrategy.GREEDY
)

# 加权随机采样：按概率分布采样
text = model.generate_text(
    seed=["今", "天"],
    sampling_strategy=SamplingStrategy.WEIGHTED_RANDOM
)
```

## 使用自定义配置

```python
from config import Config

# 从字典创建配置
config_dict = {
    "model": {
        "n": 3,
        "smoothing_method": "add_k",
        "smoothing_k": 0.3
    },
    "training": {
        "random_seed": 123,
        "train_ratio": 0.75
    },
    "generation": {
        "max_length": 15,
        "sampling_strategy": "greedy"
    },
    "log": {
        "log_level": "DEBUG",
        "log_to_file": True,
        "log_file_path": "ngram.log"
    }
}

config = Config.from_dict(config_dict)
model = NGramModel(config=config.model, log_config=config.log)
```

## 运行测试

```bash
python -m pytest test_ngram.py -v
```

或直接运行：

```bash
python test_ngram.py
```

## 运行演示

```bash
python demo.py
```

## 异常处理

模块提供以下自定义异常：

| 异常类 | 说明 |
|--------|------|
| NGramError | 基础异常类 |
| EmptyCorpusError | 语料库为空 |
| InvalidContextError | 无效上下文 |
| InvalidInputTypeError | 输入类型错误 |
| ModelNotTrainedError | 模型未训练 |
| DivisionByZeroError | 除零错误 |

示例：

```python
from utils import EmptyCorpusError, ModelNotTrainedError

try:
    model.train([])
except EmptyCorpusError as e:
    print(f"错误: {e}")

try:
    model.predict_next(["今"])
except ModelNotTrainedError as e:
    print(f"错误: {e}")
```

## 重构前后核心差异

### 1. 模块化拆分

| 重构前 | 重构后 |
|--------|--------|
| 单文件包含所有逻辑 | 分离为 config、utils、model、demo 四个模块 |
| 功能耦合度高 | 职责单一，易于维护 |

### 2. 配置管理

| 重构前 | 重构后 |
|--------|--------|
| 硬编码参数（n=2, seed=42） | 配置类统一管理 |
| 参数分散在代码各处 | 支持从字典创建配置 |

### 3. 异常处理

| 重构前 | 重构后 |
|--------|--------|
| 无输入校验 | 完善的输入校验和自定义异常 |
| 潜在除零风险 | 困惑度计算时处理边界情况 |

### 4. 日志体系

| 重构前 | 重构后 |
|--------|--------|
| 使用 print 输出 | 使用 logging 模块 |
| 无法配置日志级别 | 支持多级别、文件输出 |

### 5. 扩展性

| 重构前 | 重构后 |
|--------|--------|
| 仅支持 Add-1 平滑 | 抽象平滑基类，支持多种平滑方法 |
| 仅支持加权随机采样 | 支持贪婪和加权随机两种策略 |

### 6. 类型提示

| 重构前 | 重构后 |
|--------|--------|
| 部分类型提示 | 所有方法完整的类型提示 |
| 返回值类型不明确 | 精准的返回值类型 |

### 7. 测试友好

| 重构前 | 重构后 |
|--------|--------|
| 无单元测试 | 完整的单元测试覆盖 |
| 难以单独测试方法 | 核心方法无副作用，易于测试 |

## 优化收益

1. **可维护性提升**：模块化拆分后，各模块职责清晰，修改影响范围可控
2. **可扩展性增强**：平滑方法和采样策略可轻松扩展
3. **可读性改善**：完善的文档字符串和类型提示
4. **健壮性提高**：完善的异常处理和输入校验
5. **测试友好**：核心方法可独立测试，测试覆盖率高
6. **配置灵活**：支持多种配置方式，便于实验和调参
