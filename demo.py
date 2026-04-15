#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 模型演示模块
包含完整的训练、预测、生成、评估演示流程
"""

import random
import logging

from config import (
    Config, 
    ModelConfig, 
    TrainingConfig, 
    GenerationConfig,
    LogConfig,
    SmoothingMethod,
    SamplingStrategy
)
from ngram_model import NGramModel
from utils import generate_corpus, split_corpus, setup_logger


def run_demo(config: Config = None) -> None:
    """
    运行 N-Gram 模型演示
    
    Args:
        config: 配置对象，如果为 None 则使用默认配置
    """
    if config is None:
        config = Config()
    
    logger = setup_logger("demo", config.log)
    
    logger.info("=" * 60)
    logger.info("N-Gram 语言模型演示")
    logger.info("=" * 60)
    
    random.seed(config.training.random_seed)
    
    logger.info("\n【1. 数据生成】")
    corpus = generate_corpus()
    logger.info(f"生成语料库共 {len(corpus)} 个句子")
    logger.info(f"示例句子: {''.join(corpus[0])}")
    
    train_corpus, test_corpus = split_corpus(
        corpus, 
        train_ratio=config.training.train_ratio,
        random_seed=config.training.random_seed
    )
    logger.info(f"训练集: {len(train_corpus)} 个句子")
    logger.info(f"测试集: {len(test_corpus)} 个句子")
    
    for n in [2, 3, 4]:
        logger.info("\n" + "=" * 60)
        logger.info(f"【{n}-Gram 模型】")
        logger.info("=" * 60)
        
        model_config = ModelConfig(
            n=n,
            smoothing_method=config.model.smoothing_method,
            smoothing_k=config.model.smoothing_k
        )
        
        model = NGramModel(config=model_config, log_config=config.log)
        model.train(train_corpus)
        
        logger.info(f"\n【3. 预测下一个词】")
        test_contexts = [
            ["今", "天"],
            ["天", "气"],
            ["明", "天"],
            ["早", "上"]
        ]
        
        for context in test_contexts:
            next_word, prob = model.predict_next(context)
            context_str = ''.join(context)
            logger.info(f"  上下文 '{context_str}' -> 预测: '{next_word}' (概率: {prob:.6f})")
        
        logger.info(f"\n【4. 文本生成】")
        logger.info(f"生成 5 段样本文本（最大长度 15）:")
        for i in range(5):
            seed = random.choice(train_corpus)[:2] if random.random() > 0.5 else None
            generated = model.generate_text(
                seed=seed, 
                max_length=15,
                sampling_strategy=config.generation.sampling_strategy
            )
            logger.info(f"  样本 {i+1}: {generated}")
        
        logger.info(f"\n【5. 模型评估】")
        perplexity = model.calculate_perplexity(test_corpus)
        logger.info(f"测试集困惑度 (Perplexity): {perplexity:.4f}")
        logger.info(f"解释: 困惑度越低，模型性能越好")
        logger.info(f"      困惑度为 {perplexity:.2f} 表示模型相当于面对一个")
        logger.info(f"      有 {perplexity:.2f} 个等概率选择的困惑")
    
    logger.info("\n" + "=" * 60)
    logger.info("演示完成！")
    logger.info("=" * 60)


def demo_smoothing_methods() -> None:
    """演示不同平滑方法的效果"""
    log_config = LogConfig(log_level="INFO")
    logger = setup_logger("smoothing_demo", log_config)
    
    logger.info("=" * 60)
    logger.info("平滑方法对比演示")
    logger.info("=" * 60)
    
    random.seed(42)
    corpus = generate_corpus()
    train_corpus, test_corpus = split_corpus(corpus, train_ratio=0.8, random_seed=42)
    
    smoothing_configs = [
        (SmoothingMethod.ADD_ONE, 1.0, "Add-1 (Laplace) 平滑"),
        (SmoothingMethod.ADD_K, 0.5, "Add-0.5 平滑"),
        (SmoothingMethod.ADD_K, 0.1, "Add-0.1 平滑"),
    ]
    
    for method, k, description in smoothing_configs:
        logger.info(f"\n【{description}】")
        
        model_config = ModelConfig(n=2, smoothing_method=method, smoothing_k=k)
        model = NGramModel(config=model_config, log_config=log_config)
        model.train(train_corpus)
        
        perplexity = model.calculate_perplexity(test_corpus)
        logger.info(f"测试集困惑度: {perplexity:.4f}")
        
        context = ["今", "天"]
        next_word, prob = model.predict_next(context)
        logger.info(f"上下文 '{''.join(context)}' -> 预测: '{next_word}' (概率: {prob:.6f})")


def demo_sampling_strategies() -> None:
    """演示不同采样策略的效果"""
    log_config = LogConfig(log_level="INFO")
    logger = setup_logger("sampling_demo", log_config)
    
    logger.info("=" * 60)
    logger.info("采样策略对比演示")
    logger.info("=" * 60)
    
    random.seed(42)
    corpus = generate_corpus()
    train_corpus, _ = split_corpus(corpus, train_ratio=0.8, random_seed=42)
    
    model_config = ModelConfig(n=2)
    model = NGramModel(config=model_config, log_config=log_config)
    model.train(train_corpus)
    
    seed = ["今", "天"]
    
    logger.info(f"\n【贪婪采样】")
    for i in range(3):
        generated = model.generate_text(
            seed=seed, 
            max_length=10,
            sampling_strategy=SamplingStrategy.GREEDY
        )
        logger.info(f"  样本 {i+1}: {generated}")
    
    logger.info(f"\n【加权随机采样】")
    for i in range(3):
        generated = model.generate_text(
            seed=seed, 
            max_length=10,
            sampling_strategy=SamplingStrategy.WEIGHTED_RANDOM
        )
        logger.info(f"  样本 {i+1}: {generated}")


def demo_custom_config() -> None:
    """演示自定义配置"""
    log_config = LogConfig(log_level="INFO")
    logger = setup_logger("custom_demo", log_config)
    
    logger.info("=" * 60)
    logger.info("自定义配置演示")
    logger.info("=" * 60)
    
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
            "log_level": "INFO",
            "log_to_file": False
        }
    }
    
    config = Config.from_dict(config_dict)
    logger.info(f"配置信息: {config.to_dict()}")
    
    random.seed(config.training.random_seed)
    corpus = generate_corpus()
    train_corpus, test_corpus = split_corpus(
        corpus, 
        train_ratio=config.training.train_ratio,
        random_seed=config.training.random_seed
    )
    
    model = NGramModel(config=config.model, log_config=config.log)
    model.train(train_corpus)
    
    logger.info(f"\n【生成文本】")
    for i in range(3):
        generated = model.generate_text(
            max_length=config.generation.max_length,
            sampling_strategy=config.generation.sampling_strategy
        )
        logger.info(f"  样本 {i+1}: {generated}")
    
    perplexity = model.calculate_perplexity(test_corpus)
    logger.info(f"\n测试集困惑度: {perplexity:.4f}")


def main():
    """主函数"""
    run_demo()
    
    print("\n" + "=" * 60)
    print("扩展演示")
    print("=" * 60)
    
    demo_smoothing_methods()
    demo_sampling_strategies()
    demo_custom_config()


if __name__ == "__main__":
    main()
