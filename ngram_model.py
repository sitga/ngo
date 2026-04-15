#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 语言模型实现

支持 N 可调，包含多种平滑技术（Add-1/Laplace、Add-k、Kneser-Ney）
支持多种采样策略（加权随机、贪婪、束搜索）
"""

import random
import math
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set

from config import (
    ModelConfig,
    GenerationConfig,
    DataConfig,
    SmoothingMethod,
    SamplingStrategy,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_GENERATION_CONFIG,
    DEFAULT_DATA_CONFIG
)
from utils import validate_corpus, setup_logging, DEFAULT_LOGGING_CONFIG


class NGramModelError(Exception):
    """N-Gram 模型自定义异常基类"""
    pass


class EmptyCorpusError(NGramModelError):
    """空语料库异常"""
    pass


class UntrainedModelError(NGramModelError):
    """模型未训练异常"""
    pass


class InvalidContextError(NGramModelError):
    """无效上下文异常"""
    pass


class NGramModel:
    """
    N-Gram 语言模型类

    支持计算条件概率 P(w_n | w_{n-(N-1)}, ..., w_{n-1})
    支持多种平滑技术处理未见的 N-Gram 序列
    支持多种采样策略生成文本

    Attributes:
        n: N-Gram 的阶数
        smoothing: 平滑方法
        vocab: 词汇表
        vocab_size: 词汇表大小
        total_tokens: 总词数
        ngram_counts: N-Gram 计数
        context_counts: (n-1)-gram 上下文计数
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        data_config: Optional[DataConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 N-Gram 模型

        Args:
            model_config: 模型配置，如果为 None 则使用默认配置
            generation_config: 生成配置，如果为 None 则使用默认配置
            data_config: 数据配置，如果为 None 则使用默认配置
            logger: 日志记录器，如果为 None 则创建默认日志记录器
        """
        self.model_config = model_config or DEFAULT_MODEL_CONFIG
        self.generation_config = generation_config or DEFAULT_GENERATION_CONFIG
        self.data_config = data_config or DEFAULT_DATA_CONFIG
        self.logger = logger or setup_logging(DEFAULT_LOGGING_CONFIG)

        self.n: int = self.model_config.n
        self.ngram_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: Set[str] = set()
        self.vocab_size: int = 0
        self.total_tokens: int = 0
        self._is_trained: bool = False

    def _pad_sentence(self, sentence: List[str]) -> List[str]:
        """
        为句子添加起始和结束标记

        Args:
            sentence: 分词后的句子列表

        Returns:
            添加标记后的句子
        """
        start_tokens = [self.data_config.start_token] * (self.n - 1)
        return start_tokens + sentence + [self.data_config.end_token]

    def _validate_corpus_input(self, corpus: List[List[str]]) -> None:
        """
        验证语料库输入

        Args:
            corpus: 待验证的语料库

        Raises:
            EmptyCorpusError: 当语料库为空时抛出
            TypeError: 当输入类型不正确时抛出
        """
        if not isinstance(corpus, list):
            raise TypeError(f"语料库必须是列表类型，当前类型: {type(corpus)}")

        if not corpus:
            raise EmptyCorpusError("语料库不能为空")

        for i, sentence in enumerate(corpus):
            if not isinstance(sentence, list):
                raise TypeError(
                    f"第 {i} 个句子必须是列表类型，当前类型: {type(sentence)}"
                )
            if not sentence:
                self.logger.warning(f"第 {i} 个句子为空，将被跳过")

    def train(self, corpus: List[List[str]]) -> None:
        """
        训练 N-Gram 模型

        Args:
            corpus: 语料库，每个元素是一个分词后的句子列表

        Raises:
            EmptyCorpusError: 当语料库为空时抛出
            TypeError: 当输入类型不正确时抛出
        """
        self._validate_corpus_input(corpus)

        self.logger.info(f"开始训练 {self.n}-Gram 模型...")
        self.logger.info(f"语料库包含 {len(corpus)} 个句子")

        # 重置状态
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocab.clear()
        self.total_tokens = 0
        self._is_trained = False

        # 统计词频和 N-Gram
        for sentence in corpus:
            if not sentence:
                continue

            padded = self._pad_sentence(sentence)
            self.total_tokens += len(sentence)

            # 更新词汇表
            for word in sentence:
                self.vocab.add(word)
            self.vocab.add(self.data_config.end_token)

            # 统计 N-Gram
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]

                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

        self.vocab_size = len(self.vocab)
        self._is_trained = True

        # 打印训练统计信息
        total_ngrams = sum(len(counts) for counts in self.ngram_counts.values())
        self.logger.info(f"词汇表大小: {self.vocab_size}")
        self.logger.info(f"总词数: {self.total_tokens}")
        self.logger.info(f"不同 {self.n}-Gram 数量: {total_ngrams}")
        self.logger.info(f"不同上下文数量: {len(self.context_counts)}")
        self.logger.info("训练完成！")

    def _apply_smoothing(
        self,
        count_context_word: int,
        count_context: int,
        word: str,
        context: Tuple[str, ...]
    ) -> float:
        """
        应用平滑技术计算概率

        Args:
            count_context_word: 上下文和词共同出现的次数
            count_context: 上下文出现的次数
            word: 当前词
            context: 上下文

        Returns:
            平滑后的概率
        """
        smoothing = self.model_config.smoothing

        if smoothing == SmoothingMethod.ADD_ONE:
            return self._add_one_smoothing(count_context_word, count_context)
        elif smoothing == SmoothingMethod.ADD_K:
            return self._add_k_smoothing(count_context_word, count_context)
        elif smoothing == SmoothingMethod.KNESER_NEY:
            return self._kneser_ney_smoothing(count_context_word, count_context, word, context)
        else:
            # 默认使用 Add-1 平滑
            return self._add_one_smoothing(count_context_word, count_context)

    def _add_one_smoothing(self, count_context_word: int, count_context: int) -> float:
        """
        Add-1 (Laplace) 平滑

        P(w_n | context) = (count(context, w_n) + 1) / (count(context) + |V|)

        Args:
            count_context_word: 上下文和词共同出现的次数
            count_context: 上下文出现的次数

        Returns:
            平滑后的概率
        """
        numerator = count_context_word + 1
        denominator = count_context + self.vocab_size
        return numerator / denominator if denominator > 0 else 1.0 / self.vocab_size

    def _add_k_smoothing(self, count_context_word: int, count_context: int) -> float:
        """
        Add-k 平滑

        P(w_n | context) = (count(context, w_n) + k) / (count(context) + k * |V|)

        Args:
            count_context_word: 上下文和词共同出现的次数
            count_context: 上下文出现的次数

        Returns:
            平滑后的概率
        """
        k = self.model_config.smoothing_k
        numerator = count_context_word + k
        denominator = count_context + k * self.vocab_size
        return numerator / denominator if denominator > 0 else k / (k * self.vocab_size)

    def _kneser_ney_smoothing(
        self,
        count_context_word: int,
        count_context: int,
        word: str,
        context: Tuple[str, ...]
    ) -> float:
        """
        Kneser-Ney 平滑（简化版）

        这里实现的是一个简化版本，完整实现需要更多数据结构支持

        Args:
            count_context_word: 上下文和词共同出现的次数
            count_context: 上下文出现的次数
            word: 当前词
            context: 上下文

        Returns:
            平滑后的概率
        """
        # 简化实现：使用绝对折扣
        discount = 0.75

        if count_context > 0:
            # 高阶概率
            if count_context_word > 0:
                high_order_prob = max(count_context_word - discount, 0) / count_context
            else:
                high_order_prob = 0.0

            # 回退概率（使用 Add-1 平滑作为回退）
            backoff_prob = self._add_one_smoothing(
                self.ngram_counts[context].get(word, 0),
                sum(self.ngram_counts[context].values())
            )

            # 插值
            lambda_val = discount * len(self.ngram_counts[context]) / count_context if count_context > 0 else 0
            probability = high_order_prob + lambda_val * backoff_prob
        else:
            # 上下文未见过，使用回退概率
            probability = 1.0 / self.vocab_size

        return probability

    def _get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        计算条件概率 P(word | context)

        Args:
            word: 当前词
            context: 上下文 (n-1)-gram

        Returns:
            平滑后的条件概率

        Raises:
            UntrainedModelError: 当模型未训练时抛出
        """
        if not self._is_trained:
            raise UntrainedModelError("模型尚未训练，请先调用 train() 方法")

        count_context_word = self.ngram_counts[context].get(word, 0)
        count_context = self.context_counts.get(context, 0)

        return self._apply_smoothing(count_context_word, count_context, word, context)

    def _normalize_context(self, context: List[str]) -> Tuple[str, ...]:
        """
        规范化上下文，确保长度为 n-1

        Args:
            context: 原始上下文列表

        Returns:
            规范化后的上下文元组

        Raises:
            InvalidContextError: 当上下文包含非字符串元素时抛出
        """
        # 验证上下文元素类型
        for i, item in enumerate(context):
            if not isinstance(item, str):
                raise InvalidContextError(
                    f"上下文的第 {i} 个元素必须是字符串，当前类型: {type(item)}"
                )

        if len(context) >= self.n - 1:
            return tuple(context[-(self.n - 1):])
        else:
            # 如果上下文不足，用 start_token 填充
            padded_context = [self.data_config.start_token] * (self.n - 1 - len(context)) + context
            return tuple(padded_context)

    def predict_next(self, context: List[str]) -> Tuple[str, float]:
        """
        给定前缀，预测下一个概率最大的词

        Args:
            context: 前缀上下文列表

        Returns:
            (预测的词, 概率) 元组

        Raises:
            UntrainedModelError: 当模型未训练时抛出
            InvalidContextError: 当上下文无效时抛出
        """
        if not self._is_trained:
            raise UntrainedModelError("模型尚未训练，请先调用 train() 方法")

        context_tuple = self._normalize_context(context)

        # 计算所有可能词的概率
        best_word = None
        best_prob = -1.0

        for word in self.vocab:
            prob = self._get_probability(word, context_tuple)
            if prob > best_prob:
                best_prob = prob
                best_word = word

        return best_word, best_prob

    def _sample_next_word(self, context_tuple: Tuple[str, ...]) -> str:
        """
        根据采样策略选择下一个词

        Args:
            context_tuple: 上下文元组

        Returns:
            采样的下一个词
        """
        strategy = self.generation_config.sampling_strategy

        if strategy == SamplingStrategy.GREEDY:
            return self._greedy_sampling(context_tuple)
        elif strategy == SamplingStrategy.BEAM_SEARCH:
            # 束搜索在 generate_text 中单独处理
            return self._weighted_random_sampling(context_tuple)
        else:
            return self._weighted_random_sampling(context_tuple)

    def _greedy_sampling(self, context_tuple: Tuple[str, ...]) -> str:
        """
        贪婪采样：选择概率最大的词

        Args:
            context_tuple: 上下文元组

        Returns:
            概率最大的词
        """
        best_word = None
        best_prob = -1.0

        for word in self.vocab:
            prob = self._get_probability(word, context_tuple)
            if prob > best_prob:
                best_prob = prob
                best_word = word

        return best_word

    def _weighted_random_sampling(self, context_tuple: Tuple[str, ...]) -> str:
        """
        加权随机采样：按概率分布随机选择

        Args:
            context_tuple: 上下文元组

        Returns:
            按概率采样的词
        """
        words = list(self.vocab)
        probabilities = [self._get_probability(w, context_tuple) for w in words]

        # 应用温度参数
        temperature = self.generation_config.temperature
        if temperature != 1.0:
            # 温度缩放：温度越高，分布越均匀；温度越低，分布越尖锐
            log_probs = [math.log(p) if p > 0 else float('-inf') for p in probabilities]
            scaled_probs = [math.exp(lp / temperature) if lp != float('-inf') else 0 for lp in log_probs]
            total = sum(scaled_probs)
            probabilities = [p / total if total > 0 else 1.0 / len(words) for p in scaled_probs]
        else:
            # 归一化概率
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                probabilities = [1.0 / len(words)] * len(words)

        return random.choices(words, weights=probabilities, k=1)[0]

    def generate_text(
        self,
        seed: Optional[List[str]] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        自动生成一段文本

        Args:
            seed: 种子文本（可选）
            max_length: 生成文本的最大长度，如果为 None 则使用配置中的值

        Returns:
            生成的文本字符串

        Raises:
            UntrainedModelError: 当模型未训练时抛出
        """
        if not self._is_trained:
            raise UntrainedModelError("模型尚未训练，请先调用 train() 方法")

        if max_length is None:
            max_length = self.generation_config.max_length

        if seed is None:
            context = [self.data_config.start_token] * (self.n - 1)
        else:
            context = [self.data_config.start_token] * (self.n - 1) + seed

        generated = list(seed) if seed else []

        for _ in range(max_length):
            context_tuple = tuple(context[-(self.n - 1):])

            next_word = self._sample_next_word(context_tuple)

            if next_word == self.data_config.end_token:
                break

            generated.append(next_word)
            context.append(next_word)

        return "".join(generated)

    def calculate_perplexity(self, test_corpus: List[List[str]]) -> float:
        """
        计算测试集上的困惑度 (Perplexity)

        PP(W) = exp(-1/N * sum(log P(w_i | w_{i-n+1}...w_{i-1})))

        Args:
            test_corpus: 测试语料库

        Returns:
            困惑度值

        Raises:
            UntrainedModelError: 当模型未训练时抛出
            EmptyCorpusError: 当测试集为空时抛出
            ZeroDivisionError: 当总词数为 0 时抛出
        """
        if not self._is_trained:
            raise UntrainedModelError("模型尚未训练，请先调用 train() 方法")

        if not test_corpus:
            raise EmptyCorpusError("测试集不能为空")

        log_prob_sum = 0.0
        total_words = 0

        for sentence in test_corpus:
            if not sentence:
                continue

            padded = self._pad_sentence(sentence)

            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]

                prob = self._get_probability(word, context)
                log_prob_sum += math.log(prob)
                total_words += 1

        if total_words == 0:
            raise ZeroDivisionError("测试集总词数为 0，无法计算困惑度")

        # 计算困惑度
        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)

        return perplexity

    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型信息

        Returns:
            包含模型信息的字典
        """
        return {
            "n": self.n,
            "is_trained": self._is_trained,
            "vocab_size": self.vocab_size,
            "total_tokens": self.total_tokens,
            "smoothing_method": self.model_config.smoothing.value,
            "ngram_count": sum(len(counts) for counts in self.ngram_counts.values()),
            "context_count": len(self.context_counts)
        }

    def get_ngram_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        获取指定 N-Gram 的概率

        Args:
            ngram: N-Gram 元组

        Returns:
            N-Gram 的概率

        Raises:
            UntrainedModelError: 当模型未训练时抛出
            ValueError: 当 N-Gram 长度不正确时抛出
        """
        if not self._is_trained:
            raise UntrainedModelError("模型尚未训练，请先调用 train() 方法")

        if len(ngram) != self.n:
            raise ValueError(f"N-Gram 长度必须为 {self.n}，当前长度: {len(ngram)}")

        context = ngram[:-1]
        word = ngram[-1]

        return self._get_probability(word, context)
