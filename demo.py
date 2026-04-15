#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 语言模型演示脚本

演示模型的训练、预测、文本生成和困惑度计算功能
"""

import random

from config import (
    ModelConfig,
    GenerationConfig,
    DataConfig,
    LoggingConfig,
    SmoothingMethod,
    SamplingStrategy,
    DEFAULT_DATA_CONFIG
)
from utils import (
    generate_corpus,
    split_corpus,
    setup_logging,
    get_corpus_stats
)
from ngram_model import NGramModel


def demo_basic_training():
    """演示基本训练流程"""
    print("\n" + "=" * 60)
    print("【演示 1: 基本训练流程】")
    print("=" * 60)

    # 配置日志
    logging_config = LoggingConfig(level="INFO")
    logger = setup_logging(logging_config)

    # 生成语料库
    logger.info("生成语料库...")
    corpus = generate_corpus()
    stats = get_corpus_stats(corpus)
    logger.info(f"语料库统计: {stats}")

    # 划分训练集和测试集
    train_corpus, test_corpus = split_corpus(corpus)
    logger.info(f"训练集大小: {len(train_corpus)} 句子")
    logger.info(f"测试集大小: {len(test_corpus)} 句子")

    # 创建并训练模型
    model_config = ModelConfig(n=2, smoothing=SmoothingMethod.ADD_ONE)
    model = NGramModel(model_config=model_config, logger=logger)
    model.train(train_corpus)

    return model, train_corpus, test_corpus


def demo_prediction(model: NGramModel):
    """演示下一个词预测"""
    print("\n" + "=" * 60)
    print("【演示 2: 下一个词预测】")
    print("=" * 60)

    test_contexts = [
        ["今", "天"],
        ["天", "气"],
        ["明", "天"],
        ["早", "上"]
    ]

    for context in test_contexts:
        next_word, prob = model.predict_next(context)
        context_str = "".join(context)
        print(f"  上下文 '{context_str}' -> 预测: '{next_word}' (概率: {prob:.6f})")


def demo_text_generation(model: NGramModel, train_corpus: list):
    """演示文本生成"""
    print("\n" + "=" * 60)
    print("【演示 3: 文本生成】")
    print("=" * 60)

    # 加权随机采样
    print("\n加权随机采样生成:")
    model.generation_config.sampling_strategy = SamplingStrategy.WEIGHTED_RANDOM
    for i in range(3):
        seed = random.choice(train_corpus)[:2] if random.random() > 0.5 else None
        generated = model.generate_text(seed=seed, max_length=15)
        seed_str = "".join(seed) if seed else "(无)"
        print(f"  样本 {i+1} (种子: {seed_str}): {generated}")

    # 贪婪采样
    print("\n贪婪采样生成:")
    model.generation_config.sampling_strategy = SamplingStrategy.GREEDY
    for i in range(3):
        seed = random.choice(train_corpus)[:2] if random.random() > 0.5 else None
        generated = model.generate_text(seed=seed, max_length=15)
        seed_str = "".join(seed) if seed else "(无)"
        print(f"  样本 {i+1} (种子: {seed_str}): {generated}")


def demo_perplexity_evaluation(model: NGramModel, test_corpus: list):
    """演示困惑度评估"""
    print("\n" + "=" * 60)
    print("【演示 4: 困惑度评估】")
    print("=" * 60)

    perplexity = model.calculate_perplexity(test_corpus)
    print(f"测试集困惑度 (Perplexity): {perplexity:.4f}")
    print(f"解释: 困惑度越低，模型性能越好")
    print(f"      困惑度为 {perplexity:.2f} 表示模型相当于面对一个")
    print(f"      有 {perplexity:.2f} 个等概率选择的困惑")


def demo_different_n_values():
    """演示不同 N 值的模型"""
    print("\n" + "=" * 60)
    print("【演示 5: 不同 N 值的模型比较】")
    print("=" * 60)

    # 配置日志
    logging_config = LoggingConfig(level="WARNING")  # 减少日志输出
    logger = setup_logging(logging_config)

    # 生成语料库
    corpus = generate_corpus()
    train_corpus, test_corpus = split_corpus(corpus)

    results = []

    for n in [2, 3, 4]:
        print(f"\n--- {n}-Gram 模型 ---")

        model_config = ModelConfig(n=n, smoothing=SmoothingMethod.ADD_ONE)
        model = NGramModel(model_config=model_config, logger=logger)
        model.train(train_corpus)

        # 生成文本示例
        generated = model.generate_text(max_length=15)
        print(f"生成文本: {generated}")

        # 计算困惑度
        perplexity = model.calculate_perplexity(test_corpus)
        print(f"困惑度: {perplexity:.4f}")

        results.append((n, perplexity))

    print("\n不同 N 值的困惑度比较:")
    for n, perplexity in results:
        print(f"  {n}-Gram: {perplexity:.4f}")


def demo_different_smoothing():
    """演示不同平滑方法"""
    print("\n" + "=" * 60)
    print("【演示 6: 不同平滑方法比较】")
    print("=" * 60)

    # 配置日志
    logging_config = LoggingConfig(level="WARNING")
    logger = setup_logging(logging_config)

    # 生成语料库
    corpus = generate_corpus()
    train_corpus, test_corpus = split_corpus(corpus)

    smoothing_methods = [
        (SmoothingMethod.ADD_ONE, "Add-1 (Laplace)"),
        (SmoothingMethod.ADD_K, "Add-k (k=0.5)"),
    ]

    results = []

    for method, name in smoothing_methods:
        print(f"\n--- {name} 平滑 ---")

        if method == SmoothingMethod.ADD_K:
            model_config = ModelConfig(n=2, smoothing=method, smoothing_k=0.5)
        else:
            model_config = ModelConfig(n=2, smoothing=method)

        model = NGramModel(model_config=model_config, logger=logger)
        model.train(train_corpus)

        # 计算困惑度
        perplexity = model.calculate_perplexity(test_corpus)
        print(f"困惑度: {perplexity:.4f}")

        results.append((name, perplexity))

    print("\n不同平滑方法的困惑度比较:")
    for name, perplexity in results:
        print(f"  {name}: {perplexity:.4f}")


def demo_temperature_effect():
    """演示温度参数对生成的影响"""
    print("\n" + "=" * 60)
    print("【演示 7: 温度参数效果】")
    print("=" * 60)

    # 配置日志
    logging_config = LoggingConfig(level="WARNING")
    logger = setup_logging(logging_config)

    # 生成语料库
    corpus = generate_corpus()
    train_corpus, _ = split_corpus(corpus)

    model_config = ModelConfig(n=2)
    generation_config = GenerationConfig(
        sampling_strategy=SamplingStrategy.WEIGHTED_RANDOM,
        max_length=20
    )
    model = NGramModel(
        model_config=model_config,
        generation_config=generation_config,
        logger=logger
    )
    model.train(train_corpus)

    temperatures = [0.5, 1.0, 2.0]

    for temp in temperatures:
        print(f"\n--- 温度 = {temp} ---")
        model.generation_config.temperature = temp

        for i in range(3):
            generated = model.generate_text(max_length=15)
            print(f"  样本 {i+1}: {generated}")


def demo_model_info(model: NGramModel):
    """演示获取模型信息"""
    print("\n" + "=" * 60)
    print("【演示 8: 模型信息】")
    print("=" * 60)

    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


def demo_custom_corpus():
    """演示使用自定义语料库"""
    print("\n" + "=" * 60)
    print("【演示 9: 自定义语料库】")
    print("=" * 60)

    # 自定义语料库
    custom_sentences = [
        "人工智能改变世界",
        "机器学习是人工智能的分支",
        "深度学习是机器学习的一种方法",
        "神经网络模拟人脑结构",
        "自然语言处理让机器理解语言",
        "计算机视觉让机器看懂图像",
        "强化学习通过试错来学习",
        "数据是训练模型的基础",
        "算法决定模型的性能",
        "算力支撑大规模训练"
    ]

    # 分词
    custom_corpus = [list(s) for s in custom_sentences]

    logging_config = LoggingConfig(level="INFO")
    logger = setup_logging(logging_config)

    model_config = ModelConfig(n=2)
    model = NGramModel(model_config=model_config, logger=logger)
    model.train(custom_corpus)

    # 生成文本
    print("\n生成文本:")
    for i in range(5):
        generated = model.generate_text(max_length=10)
        print(f"  样本 {i+1}: {generated}")

    # 预测下一个词
    print("\n预测下一个词:")
    test_contexts = [["人", "工"], ["机", "器"], ["学", "习"]]
    for context in test_contexts:
        next_word, prob = model.predict_next(context)
        context_str = "".join(context)
        print(f"  上下文 '{context_str}' -> 预测: '{next_word}' (概率: {prob:.6f})")


def main():
    """主函数"""
    print("=" * 60)
    print("N-Gram 语言模型演示")
    print("=" * 60)

    # 设置随机种子以保证可重复性
    random.seed(DEFAULT_DATA_CONFIG.random_seed)

    # 演示 1: 基本训练
    model, train_corpus, test_corpus = demo_basic_training()

    # 演示 2: 下一个词预测
    demo_prediction(model)

    # 演示 3: 文本生成
    demo_text_generation(model, train_corpus)

    # 演示 4: 困惑度评估
    demo_perplexity_evaluation(model, test_corpus)

    # 演示 5: 不同 N 值比较
    demo_different_n_values()

    # 演示 6: 不同平滑方法
    demo_different_smoothing()

    # 演示 7: 温度参数效果
    demo_temperature_effect()

    # 演示 8: 模型信息
    demo_model_info(model)

    # 演示 9: 自定义语料库
    demo_custom_corpus()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
