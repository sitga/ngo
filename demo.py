#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示脚本 - N-Gram 语言模型使用示例
"""
import random
import logging

from config import ModelConfig, SamplingStrategy
from ngram_model import NGramModel
from utils import generate_corpus, split_corpus, setup_logger


def run_demo():
    """运行 N-Gram 模型完整演示"""
    logger = setup_logger('demo')
    logger.info("=" * 60)
    logger.info("N-Gram 语言模型演示")
    logger.info("=" * 60)
    
    random_seed = 42
    random.seed(random_seed)
    
    logger.info("\n【1. 数据生成】")
    corpus = generate_corpus()
    logger.info(f"生成语料库共 {len(corpus)} 个句子")
    logger.info(f"示例句子: {''.join(corpus[0])}")
    
    train_corpus, test_corpus = split_corpus(
        corpus, train_ratio=0.8, random_seed=random_seed
    )
    logger.info(f"训练集: {len(train_corpus)} 个句子")
    logger.info(f"测试集: {len(test_corpus)} 个句子")
    
    for n in [2, 3, 4]:
        logger.info("\n" + "=" * 60)
        logger.info(f"【{n}-Gram 模型】")
        logger.info("=" * 60)
        
        config = ModelConfig(n=n, random_seed=random_seed)
        model = NGramModel(config=config)
        model.train(train_corpus)
        
        logger.info("\n【2. 预测下一个词】")
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
        
        logger.info("\n【3. 文本生成】")
        logger.info(f"生成 5 段样本文本（最大长度 15）:")
        for i in range(5):
            seed = random.choice(train_corpus)[:2] if random.random() > 0.5 else None
            generated = model.generate_text(seed=seed, max_length=15)
            logger.info(f"  样本 {i+1}: {generated}")
        
        logger.info("\n【4. 对比采样策略】")
        logger.info("  贪婪采样 vs 加权随机采样:")
        test_seed = ["今", "天"]
        greedy_text = model.generate_text(
            seed=test_seed, max_length=10, strategy=SamplingStrategy.GREEDY
        )
        random_text = model.generate_text(
            seed=test_seed, max_length=10, strategy=SamplingStrategy.WEIGHTED_RANDOM
        )
        logger.info(f"    贪婪采样: {greedy_text}")
        logger.info(f"    随机采样: {random_text}")
        
        logger.info("\n【5. 模型评估】")
        perplexity = model.calculate_perplexity(test_corpus)
        logger.info(f"测试集困惑度 (Perplexity): {perplexity:.4f}")
        logger.info(f"解释: 困惑度越低，模型性能越好")
        logger.info(f"      困惑度为 {perplexity:.2f} 表示模型相当于面对一个")
        logger.info(f"      有 {perplexity:.2f} 个等概率选择的困惑")
    
    logger.info("\n" + "=" * 60)
    logger.info("演示完成！")
    logger.info("=" * 60)


def quick_start_example():
    """快速开始示例"""
    print("\n=== 快速开始示例 ===")
    
    corpus = [
        list("我喜欢吃苹果"),
        list("我喜欢吃香蕉"),
        list("今天天气真好")
    ]
    
    model = NGramModel(ModelConfig(n=2, random_seed=42))
    model.train(corpus)
    
    print("\n1. 预测下一个词:")
    next_word, prob = model.predict_next(["我", "喜"])
    print(f"   上下文 '我喜' -> '{next_word}' (概率: {prob:.4f})")
    
    print("\n2. 生成文本:")
    text = model.generate_text(seed=["今", "天"], max_length=10)
    print(f"   生成结果: {text}")
    
    print("\n3. 计算困惑度:")
    test_data = [list("我喜欢吃橙子")]
    ppl = model.calculate_perplexity(test_data)
    print(f"   困惑度: {ppl:.4f}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    run_demo()
    quick_start_example()
