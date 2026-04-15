#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块

包含语料生成、数据集划分、日志配置等工具函数
"""

import random
import logging
from typing import List, Tuple, Optional

from config import (
    DataConfig,
    LoggingConfig,
    CorpusConfig,
    DEFAULT_DATA_CONFIG,
    DEFAULT_LOGGING_CONFIG,
    DEFAULT_CORPUS_CONFIG
)


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    配置日志系统

    Args:
        config: 日志配置，如果为 None 则使用默认配置

    Returns:
        配置好的 Logger 实例
    """
    if config is None:
        config = DEFAULT_LOGGING_CONFIG

    logger = logging.getLogger("ngram_model")
    logger.setLevel(getattr(logging, config.level.upper()))

    # 清除已有的 handlers
    logger.handlers.clear()

    formatter = logging.Formatter(config.format)

    # 控制台输出
    if config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if config.file_path:
        file_handler = logging.FileHandler(config.file_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def generate_corpus(
    corpus_config: Optional[CorpusConfig] = None,
    data_config: Optional[DataConfig] = None
) -> List[List[str]]:
    """
    生成模拟语料库

    Args:
        corpus_config: 语料库配置，如果为 None 则使用默认配置
        data_config: 数据配置（用于获取随机种子），如果为 None 则使用默认配置

    Returns:
        分词后的语料库列表，每个句子是一个字符列表

    Raises:
        ValueError: 当语料库为空时抛出
    """
    if corpus_config is None:
        corpus_config = DEFAULT_CORPUS_CONFIG
    if data_config is None:
        data_config = DEFAULT_DATA_CONFIG

    # 设置随机种子
    random.seed(data_config.random_seed)

    sentences = corpus_config.sentences

    if not sentences:
        raise ValueError("语料库不能为空")

    # 分词（按字符分词，适用于中文）
    corpus = [list(sentence) for sentence in sentences]

    return corpus


def split_corpus(
    corpus: List[List[str]],
    data_config: Optional[DataConfig] = None
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    将语料库划分为训练集和测试集

    Args:
        corpus: 完整语料库
        data_config: 数据配置，如果为 None 则使用默认配置

    Returns:
        (训练集, 测试集) 元组

    Raises:
        ValueError: 当语料库为空时抛出
        TypeError: 当输入类型不正确时抛出
    """
    if data_config is None:
        data_config = DEFAULT_DATA_CONFIG

    # 输入校验
    if not isinstance(corpus, list):
        raise TypeError(f"语料库必须是列表类型，当前类型: {type(corpus)}")

    if not corpus:
        raise ValueError("语料库不能为空")

    # 验证每个句子都是列表
    for i, sentence in enumerate(corpus):
        if not isinstance(sentence, list):
            raise TypeError(f"第 {i} 个句子必须是列表类型，当前类型: {type(sentence)}")

    # 复制语料库以避免修改原始数据
    corpus_copy = corpus.copy()

    # 打乱语料库
    random.shuffle(corpus_copy)

    # 划分训练集和测试集
    split_idx = int(len(corpus_copy) * data_config.train_ratio)
    train_corpus = corpus_copy[:split_idx]
    test_corpus = corpus_copy[split_idx:]

    return train_corpus, test_corpus


def validate_corpus(corpus: List[List[str]]) -> None:
    """
    验证语料库格式是否正确

    Args:
        corpus: 待验证的语料库

    Raises:
        ValueError: 当语料库格式不正确时抛出
        TypeError: 当输入类型不正确时抛出
    """
    if not isinstance(corpus, list):
        raise TypeError(f"语料库必须是列表类型，当前类型: {type(corpus)}")

    if not corpus:
        raise ValueError("语料库不能为空")

    for i, sentence in enumerate(corpus):
        if not isinstance(sentence, list):
            raise TypeError(f"第 {i} 个句子必须是列表类型，当前类型: {type(sentence)}")

        if not sentence:
            raise ValueError(f"第 {i} 个句子不能为空")

        for j, word in enumerate(sentence):
            if not isinstance(word, str):
                raise TypeError(
                    f"第 {i} 个句子的第 {j} 个词必须是字符串类型，"
                    f"当前类型: {type(word)}"
                )


def tokenize_text(text: str) -> List[str]:
    """
    将文本分词（按字符分词，适用于中文）

    Args:
        text: 待分词的文本

    Returns:
        分词后的字符列表

    Raises:
        TypeError: 当输入不是字符串时抛出
        ValueError: 当输入为空字符串时抛出
    """
    if not isinstance(text, str):
        raise TypeError(f"输入必须是字符串类型，当前类型: {type(text)}")

    if not text.strip():
        raise ValueError("输入文本不能为空")

    return list(text.strip())


def detokenize_text(tokens: List[str]) -> str:
    """
    将字符列表还原为文本

    Args:
        tokens: 字符列表

    Returns:
        还原后的文本字符串

    Raises:
        TypeError: 当输入类型不正确时抛出
    """
    if not isinstance(tokens, list):
        raise TypeError(f"输入必须是列表类型，当前类型: {type(tokens)}")

    return "".join(str(token) for token in tokens)


def get_corpus_stats(corpus: List[List[str]]) -> dict:
    """
    获取语料库统计信息

    Args:
        corpus: 语料库

    Returns:
        包含统计信息的字典
    """
    validate_corpus(corpus)

    total_sentences = len(corpus)
    total_words = sum(len(sentence) for sentence in corpus)
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0

    # 统计词汇表
    vocab = set()
    for sentence in corpus:
        for word in sentence:
            vocab.add(word)

    return {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "avg_sentence_length": avg_sentence_length,
        "vocab_size": len(vocab)
    }
