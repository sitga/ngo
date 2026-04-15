# N-Gram 语言模型

一个结构清晰、可扩展的 N-Gram 语言模型实现，支持多种平滑技术和采样策略。

## 模块结构

```
ngo/
├── config.py        # 配置模块：模型参数、日志配置、枚举类型
├── utils.py         # 工具模块：异常类、日志配置、语料处理函数
├── ngram_model.py   # 核心模型：N-Gram 语言模型类
├── demo.py          # 演示脚本：完整功能展示和快速开始示例
└── test_ngram.py    # 单元测试：核心功能测试用例
```

### 各模块职责

| 文件 | 职责说明 |
|------|----------|
| **config.py** | 统一管理所有配置参数，包括平滑方式、采样策略、模型参数等 |
| **utils.py** | 工具函数集合，语料生成、数据集划分、日志配置、自定义异常 |
| **ngram_model.py** | 核心模型类，包含训练、预测、文本生成、困惑度计算等逻辑 |
| **demo.py** | 演示脚本，展示如何使用模型 |
| **test_ngram.py** | 单元测试，保证代码质量 |

## 快速开始

### 1. 基本使用

```python
from config import ModelConfig
from ngram_model import NGramModel

# 准备语料库（按字符分词）
corpus = [
    list("我喜欢吃苹果"),
    list("我喜欢吃香蕉"),
    list("今天天气真好")
]

# 创建并训练模型
config = ModelConfig(n=2, random_seed=42)
model = NGramModel(config)
model.train(corpus)
```

### 2. 预测下一个词

```python
# 给定上下文预测下一个词
next_word, prob = model.predict_next(["我", "喜"])
print(f"上下文 '我喜' -> 预测: '{next_word}' (概率: {prob:.4f})")
```

### 3. 生成文本

```python
# 无种子生成
text = model.generate_text(max_length=15)
print(f"生成文本: {text}")

# 带种子生成
text = model.generate_text(seed=["今", "天"], max_length=10)
print(f"基于种子生成: {text}")
```

### 4. 计算困惑度

```python
test_corpus = [list("我喜欢吃橙子")]
perplexity = model.calculate_perplexity(test_corpus)
print(f"困惑度: {perplexity:.4f}")
```

## 配置参数说明

### ModelConfig 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | 2 | N-Gram 阶数（必须 > 1） |
| `random_seed` | int | 42 | 随机种子，保证可重复性 |
| `train_ratio` | float | 0.8 | 训练集划分比例 |
| `max_generation_length` | int | 20 | 文本生成最大长度 |
| `smoothing_type` | SmoothingType | ADD_1 | 平滑方式 |
| `add_k` | float | 1.0 | Add-k 平滑的 k 值 |
| `sampling_strategy` | SamplingStrategy | WEIGHTED_RANDOM | 采样策略 |
| `start_token` | str | `<START>` | 句子起始标记 |
| `end_token` | str | `<END>` | 句子结束标记 |

### SmoothingType 平滑方式

| 枚举值 | 说明 |
|--------|------|
| `ADD_1` | Add-1 (Laplace) 平滑，最常用 |
| `ADD_K` | Add-k 平滑，可调整 k 值 |

### SamplingStrategy 采样策略

| 枚举值 | 说明 | 适用场景 |
|--------|------|----------|
| `GREEDY` | 贪婪采样，总是选择概率最大的词 | 确定性生成 |
| `WEIGHTED_RANDOM` | 加权随机采样，按概率分布采样 | 多样性生成 |

## 高级扩展示例

### 示例 1: 使用 Add-k 平滑

```python
from config import ModelConfig, SmoothingType

# 使用 Add-0.5 平滑
config = ModelConfig(
    n=3,
    smoothing_type=SmoothingType.ADD_K,
    add_k=0.5,
    random_seed=42
)
model = NGramModel(config)
model.train(corpus)
```

### 示例 2: 对比不同采样策略

```python
from config import SamplingStrategy

seed = ["今", "天"]

# 贪婪采样（确定性）
greedy_text = model.generate_text(
    seed=seed, max_length=10, strategy=SamplingStrategy.GREEDY
)
print(f"贪婪采样: {greedy_text}")

# 加权随机采样（多样性）
for i in range(3):
    random_text = model.generate_text(
        seed=seed, max_length=10, strategy=SamplingStrategy.WEIGHTED_RANDOM
    )
    print(f"随机采样 {i+1}: {random_text}")
```

### 示例 3: 扩展新的平滑方式

```python
# 在 ngram_model.py 的 NGramModel 类中新增方法
def _apply_kneser_ney_smoothing(
    self,
    count_context_word: int,
    count_context: int,
    discount: float = 0.75
) -> float:
    # Kneser-Ney 平滑实现
    pass

# 在 _get_probability 方法中新增分支
if self.config.smoothing_type == SmoothingType.KNESER_NEY:
    return self._apply_kneser_ney_smoothing(...)
```

### 示例 4: 扩展新的采样策略

```python
# 在 ngram_model.py 的 NGramModel 类中新增方法
def _sample_beam_search(
    self,
    context_tuple: Tuple[str, ...],
    beam_width: int = 3
) -> Tuple[str, float]:
    # 束搜索实现
    pass

# 在 predict_next 方法中新增分支
if strategy == SamplingStrategy.BEAM_SEARCH:
    return self._sample_beam_search(context_tuple)
```

## 运行演示和测试

### 运行完整演示

```bash
python demo.py
```

### 运行单元测试

```bash
python test_ngram.py
```

## 重构前后对比

### 核心差异点

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| **代码结构** | 单文件，所有代码在 `ngram_model.py` | 模块化拆分，5个文件各司其职 |
| **配置管理** | 参数硬编码分散各处 | 统一 `ModelConfig` 类集中管理 |
| **日志输出** | 使用 `print()` 输出 | 标准 `logging` 模块，支持级别和文件输出 |
| **异常处理** | 无任何异常处理 | 完整的异常层次和输入校验 |
| **平滑策略** | Add-1 硬编码在概率计算中 | 抽离为独立方法，支持扩展 |
| **采样策略** | 仅加权随机采样，硬编码 | 支持贪婪/随机两种策略，可扩展 |
| **类型提示** | 基础类型提示 | 完整精准的类型注解 |
| **可测试性** | 逻辑耦合，难以单独测试 | 方法无副作用，完整单元测试覆盖 |

### 重构收益

1. **可维护性提升**：模块化拆分，职责清晰，易于理解和修改
2. **可扩展性增强**：平滑和采样策略抽离，新增方式无需修改核心逻辑
3. **鲁棒性增强**：完善的输入校验和异常处理，避免静默失败
4. **调试效率提升**：标准日志体系，支持不同级别和文件输出
5. **测试友好**：纯函数设计，便于单元测试和回归验证
6. **开发体验**：统一配置管理，IDE 类型提示友好，文档完善

## 核心 API

### NGramModel 类

```python
class NGramModel:
    def __init__(config, logger)                # 初始化模型
    def train(corpus)                           # 训练模型
    def predict_next(context, strategy)         # 预测下一个词
    def generate_text(seed, max_length, strategy)  # 生成文本
    def calculate_perplexity(test_corpus)       # 计算困惑度
    def get_config()                            # 获取配置信息
```

## 异常层次

```
NGramError (基类)
├── EmptyCorpusError       # 空语料库异常
├── InvalidContextError    # 无效上下文异常
├── InvalidInputError      # 无效输入异常
└── VocabularyError        # 词汇表异常
```
