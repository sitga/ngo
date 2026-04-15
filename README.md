# N-Gram 语言模型

一个模块化、可扩展的 N-Gram 语言模型实现，支持多种平滑技术和采样策略。

## 模块结构

```
.
├── config.py        # 配置管理模块
├── utils.py         # 工具函数模块
├── ngram_model.py   # 核心 N-Gram 模型
├── demo.py          # 演示脚本
├── test_ngram.py    # 单元测试
└── README.md        # 项目文档
```

### 模块说明

#### `config.py` - 配置管理
- `ModelConfig`: 模型配置（N-Gram 阶数、平滑方法）
- `GenerationConfig`: 生成配置（最大长度、采样策略、温度参数）
- `DataConfig`: 数据配置（训练集比例、随机种子、特殊标记）
- `LoggingConfig`: 日志配置（日志级别、格式、输出方式）
- `CorpusConfig`: 语料库配置（预定义语料）
- `SmoothingMethod`: 平滑方法枚举（Add-1、Add-k、Kneser-Ney）
- `SamplingStrategy`: 采样策略枚举（加权随机、贪婪、束搜索）

#### `utils.py` - 工具函数
- `setup_logging()`: 配置日志系统
- `generate_corpus()`: 生成模拟语料库
- `split_corpus()`: 划分训练集和测试集
- `validate_corpus()`: 验证语料库格式
- `tokenize_text()`: 文本分词
- `detokenize_text()`: 文本还原
- `get_corpus_stats()`: 获取语料库统计信息

#### `ngram_model.py` - 核心模型
- `NGramModel`: N-Gram 语言模型类
  - `train()`: 训练模型
  - `predict_next()`: 预测下一个词
  - `generate_text()`: 生成文本
  - `calculate_perplexity()`: 计算困惑度
  - `get_model_info()`: 获取模型信息
  - `get_ngram_probability()`: 获取 N-Gram 概率
- 自定义异常类：
  - `NGramModelError`: 基类异常
  - `EmptyCorpusError`: 空语料库异常
  - `UntrainedModelError`: 未训练模型异常
  - `InvalidContextError`: 无效上下文异常

#### `demo.py` - 演示脚本
包含多个演示函数：
- `demo_basic_training()`: 基本训练流程
- `demo_prediction()`: 下一个词预测
- `demo_text_generation()`: 文本生成
- `demo_perplexity_evaluation()`: 困惑度评估
- `demo_different_n_values()`: 不同 N 值比较
- `demo_different_smoothing()`: 不同平滑方法比较
- `demo_temperature_effect()`: 温度参数效果
- `demo_custom_corpus()`: 自定义语料库

## 快速开始

### 安装依赖

本项目仅使用 Python 标准库，无需额外安装依赖。

### 运行演示

```bash
python demo.py
```

### 基本使用示例

#### 1. 训练模型

```python
from config import ModelConfig, LoggingConfig
from utils import generate_corpus, split_corpus, setup_logging
from ngram_model import NGramModel

# 配置日志
logger = setup_logging(LoggingConfig(level="INFO"))

# 生成语料库
corpus = generate_corpus()
train_corpus, test_corpus = split_corpus(corpus)

# 创建并训练模型
model_config = ModelConfig(n=2)  # Bigram
model = NGramModel(model_config=model_config, logger=logger)
model.train(train_corpus)
```

#### 2. 预测下一个词

```python
# 预测下一个词
context = ["今", "天"]
next_word, probability = model.predict_next(context)
print(f"上下文: {''.join(context)}")
print(f"预测下一个词: {next_word} (概率: {probability:.6f})")
```

#### 3. 生成文本

```python
# 生成文本
generated = model.generate_text(max_length=20)
print(f"生成文本: {generated}")

# 带种子的文本生成
seed = ["明", "天"]
generated_with_seed = model.generate_text(seed=seed, max_length=15)
print(f"带种子生成: {generated_with_seed}")
```

#### 4. 计算困惑度

```python
# 计算困惑度
perplexity = model.calculate_perplexity(test_corpus)
print(f"困惑度: {perplexity:.4f}")
```

## 配置参数说明

### ModelConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | 2 | N-Gram 阶数（>= 1） |
| `smoothing` | SmoothingMethod | ADD_ONE | 平滑方法 |
| `smoothing_k` | float | 1.0 | Add-k 平滑的参数 k（> 0） |

### GenerationConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_length` | int | 20 | 生成文本的最大长度（>= 1） |
| `sampling_strategy` | SamplingStrategy | WEIGHTED_RANDOM | 采样策略 |
| `beam_width` | int | 3 | 束搜索宽度（>= 1） |
| `temperature` | float | 1.0 | 温度参数（> 0），越高越随机 |

### DataConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_ratio` | float | 0.8 | 训练集比例（0 < ratio < 1） |
| `random_seed` | int | 42 | 随机种子 |
| `start_token` | str | "<START>" | 句子起始标记 |
| `end_token` | str | "<END>" | 句子结束标记 |

### LoggingConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | str | "INFO" | 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL） |
| `format` | str | "%(asctime)s - %(name)s - %(levelname)s - %(message)s" | 日志格式 |
| `file_path` | str | None | 日志文件路径（None 表示只输出到控制台） |
| `console_output` | bool | True | 是否输出到控制台 |

## 扩展平滑方式

### 使用 Add-k 平滑

```python
from config import ModelConfig, SmoothingMethod

model_config = ModelConfig(
    n=2,
    smoothing=SmoothingMethod.ADD_K,
    smoothing_k=0.5  # k 值
)
model = NGramModel(model_config=model_config)
```

### 使用 Kneser-Ney 平滑

```python
from config import ModelConfig, SmoothingMethod

model_config = ModelConfig(
    n=2,
    smoothing=SmoothingMethod.KNESER_NEY
)
model = NGramModel(model_config=model_config)
```

### 自定义平滑方法

可以通过继承 `NGramModel` 类并重写 `_apply_smoothing` 方法来添加自定义平滑：

```python
from ngram_model import NGramModel

class CustomNGramModel(NGramModel):
    def _apply_smoothing(self, count_context_word, count_context, word, context):
        # 自定义平滑逻辑
        # 返回平滑后的概率
        pass
```

## 扩展采样策略

### 使用贪婪采样

```python
from config import GenerationConfig, SamplingStrategy

generation_config = GenerationConfig(
    sampling_strategy=SamplingStrategy.GREEDY
)
model = NGramModel(generation_config=generation_config)
```

### 调整温度参数

```python
from config import GenerationConfig

generation_config = GenerationConfig(
    sampling_strategy=SamplingStrategy.WEIGHTED_RANDOM,
    temperature=0.5  # 较低温度使分布更尖锐
)
model = NGramModel(generation_config=generation_config)
```

### 自定义采样策略

可以通过继承 `NGramModel` 类并重写 `_sample_next_word` 方法来添加自定义采样策略：

```python
from ngram_model import NGramModel

class CustomNGramModel(NGramModel):
    def _sample_next_word(self, context_tuple):
        # 自定义采样逻辑
        # 返回采样的词
        pass
```

## 运行测试

```bash
python test_ngram.py
```

测试覆盖：
- 模型初始化和训练
- 概率计算
- 下一个词预测
- 文本生成
- 困惑度计算
- 异常处理
- 配置验证
- 工具函数
- 集成测试

## 重构前后的核心差异

### 1. 模块化拆分

**重构前**：
- 所有代码在一个文件中
- 主函数 `main()` 包含数据生成、模型训练、演示等所有逻辑

**重构后**：
- `config.py`: 集中管理所有配置
- `utils.py`: 工具函数独立
- `ngram_model.py`: 仅保留核心模型类
- `demo.py`: 演示逻辑独立

**收益**：
- 代码职责清晰，便于维护
- 模块可独立测试和复用

### 2. 配置化管理

**重构前**：
- 随机种子、默认参数硬编码
- 语料库内容硬编码在代码中

**重构后**：
- 使用 `dataclass` 定义配置类
- 配置参数可灵活调整
- 支持配置验证

**收益**：
- 无需修改代码即可调整参数
- 配置可复用、可序列化

### 3. 异常处理

**重构前**：
- 无输入校验
- 无除零异常处理
- 无类型检查

**重构后**：
- 自定义异常类体系
- 关键方法添加输入校验
- 明确的异常说明

**收益**：
- 更好的错误提示
- 更健壮的代码
- 便于调试

### 4. 日志优化

**重构前**：
- 使用 `print` 输出信息
- 无法调整日志级别

**重构后**：
- 使用 `logging` 模块
- 支持不同日志级别
- 支持控制台/文件输出

**收益**：
- 标准化日志管理
- 可配置日志输出
- 便于生产环境使用

### 5. 扩展性增强

**重构前**：
- 平滑方法硬编码
- 采样逻辑耦合在生成方法中

**重构后**：
- 平滑方法抽离为独立方法
- 采样策略可配置
- 支持温度参数

**收益**：
- 易于添加新的平滑方法
- 易于添加新的采样策略
- 更灵活的文本生成控制

### 6. 类型提示

**重构前**：
- 部分类型提示缺失
- 返回值类型不明确

**重构后**：
- 所有方法补充精准类型提示
- 使用 `Optional`、`Tuple`、`Dict` 等类型

**收益**：
- 更好的 IDE 支持
- 便于静态类型检查
- 提高代码可读性

### 7. 测试友好

**重构前**：
- 核心方法与 `print` 耦合
- 难以进行单元测试

**重构后**：
- 核心方法无副作用
- 依赖注入（日志记录器）
- 完整的单元测试覆盖

**收益**：
- 易于编写单元测试
- 保证代码质量
- 便于重构和维护

### 8. 代码规范

**重构前**：
- 部分命名不一致
- 文档字符串不完善

**重构后**：
- 遵循 PEP8 规范
- 统一的命名风格
- 完善的文档字符串

**收益**：
- 提高代码可读性
- 便于团队协作
- 生成文档更方便

## 许可证

MIT License
