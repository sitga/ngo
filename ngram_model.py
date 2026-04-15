#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 语言模型核心模块
支持 N 可调，包含多种平滑技术和采样策略
"""
import math
import random
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any

from config import ModelConfig, SmoothingType, SamplingStrategy, DEFAULT_MODEL_CONFIG
from utils import (
    setup_logger,
    validate_corpus,
    EmptyCorpusError,
    InvalidContextError,
    InvalidInputError,
    VocabularyError
)


class NGramModel:
    """
    N-Gram 语言模型类
    
    支持计算条件概率 P(w_n | w_{n-(N-1)}, ..., w_{n-1})
    支持多种平滑技术和采样策略
    
    Attributes:
        config: 模型配置对象
        n: N-Gram 阶数
        ngram_counts: n-gram 计数字典
        context_counts: (n-1)-gram 上下文计数
        vocab: 词汇表集合
        vocab_size: 词汇表大小
        total_tokens: 训练语料总词数
        logger: 日志记录器
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 N-Gram 模型
        
        Args:
            config: 模型配置对象，使用默认配置 if None
            logger: 日志记录器，自动创建 if None
            
        Raises:
            InvalidInputError: n 值小于等于 1
        """
        self.config = config or DEFAULT_MODEL_CONFIG
        self.n = self.config.n
        
        if self.n <= 1:
            raise InvalidInputError(f"N-Gram 阶数 n 必须大于 1，当前值: {self.n}")
        
        self.ngram_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: set = set()
        self.vocab_size: int = 0
        self.total_tokens: int = 0
        
        self.logger = logger or setup_logger(f"ngram_{self.n}gram")
        
        random.seed(self.config.random_seed)
    
    def _pad_sentence(self, sentence: List[str]) -> List[str]:
        """
        为句子添加起始和结束标记
        
        Args:
            sentence: 分词后的句子列表
            
        Returns:
            添加标记后的句子
        """
        padded = ([self.config.START_TOKEN] * (self.n - 1) + 
                  sentence + 
                  [self.config.END_TOKEN])
        return padded
    
    def _validate_context(self, context: Tuple[str, ...]) -> None:
        """
        验证上下文有效性
        
        Args:
            context: 上下文元组
            
        Raises:
            InvalidContextError: 上下文长度不正确
        """
        if len(context) != self.n - 1:
            raise InvalidContextError(
                f"上下文长度必须为 {self.n - 1}，当前长度: {len(context)}"
            )
    
    def _apply_add_k_smoothing(
        self,
        count_context_word: int,
        count_context: int,
        k: float = 1.0
    ) -> float:
        """
        应用 Add-k 平滑计算概率
        
        Args:
            count_context_word: (context, word) 共现次数
            count_context: context 出现次数
            k: 平滑参数，k=1 即 Add-1 (Laplace) 平滑
            
        Returns:
            平滑后的条件概率
        """
        probability = (count_context_word + k) / (count_context + k * self.vocab_size)
        return probability
    
    def _get_probability(
        self,
        word: str,
        context: Tuple[str, ...]
    ) -> float:
        """
        计算条件概率 P(word | context)
        
        Args:
            word: 当前词
            context: 上下文 (n-1)-gram 元组
            
        Returns:
            平滑后的条件概率
            
        Raises:
            VocabularyError: 词汇表未初始化
            InvalidContextError: 上下文长度错误
        """
        if self.vocab_size == 0:
            raise VocabularyError("词汇表为空，请先训练模型")
        
        self._validate_context(context)
        
        count_context_word = self.ngram_counts[context].get(word, 0)
        count_context = self.context_counts.get(context, 0)
        
        if self.config.smoothing_type == SmoothingType.ADD_1:
            return self._apply_add_k_smoothing(count_context_word, count_context, k=1.0)
        elif self.config.smoothing_type == SmoothingType.ADD_K:
            return self._apply_add_k_smoothing(
                count_context_word, count_context, k=self.config.add_k
            )
        
        return self._apply_add_k_smoothing(count_context_word, count_context, k=1.0)
    
    def _prepare_context(self, context: List[str]) -> Tuple[str, ...]:
        """
        准备和规范化上下文
        
        Args:
            context: 输入上下文列表
            
        Returns:
            处理后的上下文元组（长度为 n-1）
        """
        if len(context) >= self.n - 1:
            context_tuple = tuple(context[-(self.n - 1):])
        else:
            pad_length = self.n - 1 - len(context)
            padded_context = [self.config.START_TOKEN] * pad_length + context
            context_tuple = tuple(padded_context)
        
        return context_tuple
    
    def _sample_greedy(
        self,
        context_tuple: Tuple[str, ...]
    ) -> Tuple[str, float]:
        """
        贪婪采样：选择概率最大的词
        
        Args:
            context_tuple: 上下文元组
            
        Returns:
            (预测的词, 对应概率)
        """
        best_word = None
        best_prob = -1.0
        
        for word in self.vocab:
            prob = self._get_probability(word, context_tuple)
            if prob > best_prob:
                best_prob = prob
                best_word = word
        
        return best_word, best_prob
    
    def _sample_weighted_random(
        self,
        context_tuple: Tuple[str, ...]
    ) -> Tuple[str, float]:
        """
        加权随机采样：按概率分布采样
        
        Args:
            context_tuple: 上下文元组
            
        Returns:
            (采样的词, 对应概率)
        """
        words = list(self.vocab)
        probabilities = [self._get_probability(w, context_tuple) for w in words]
        
        total_prob = sum(probabilities)
        if total_prob == 0:
            probabilities = [1.0 / len(words)] * len(words)
        else:
            probabilities = [p / total_prob for p in probabilities]
        
        next_word = random.choices(words, weights=probabilities, k=1)[0]
        word_idx = words.index(next_word)
        word_prob = probabilities[word_idx]
        
        return next_word, word_prob
    
    def train(self, corpus: List[List[str]]) -> None:
        """
        训练 N-Gram 模型
        
        Args:
            corpus: 语料库，每个元素是一个分词后的句子列表
            
        Raises:
            EmptyCorpusError: 语料库为空
            InvalidInputError: 输入格式错误
        """
        validate_corpus(corpus)
        
        self.logger.info(f"开始训练 {self.n}-Gram 模型...")
        self.logger.info(f"语料库包含 {len(corpus)} 个句子")
        
        for sentence in corpus:
            padded = self._pad_sentence(sentence)
            self.total_tokens += len(sentence)
            
            for word in sentence:
                self.vocab.add(word)
            self.vocab.add(self.config.END_TOKEN)
            
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        
        self.vocab_size = len(self.vocab)
        
        total_ngrams = sum(len(counts) for counts in self.ngram_counts.values())
        self.logger.info(f"词汇表大小: {self.vocab_size}")
        self.logger.info(f"总词数: {self.total_tokens}")
        self.logger.info(f"不同 {self.n}-Gram 数量: {total_ngrams}")
        self.logger.info(f"不同上下文数量: {len(self.context_counts)}")
        self.logger.info("训练完成！")
    
    def predict_next(
        self,
        context: List[str],
        strategy: Optional[SamplingStrategy] = None
    ) -> Tuple[str, float]:
        """
        给定前缀，预测下一个词
        
        Args:
            context: 前缀上下文列表
            strategy: 采样策略，使用配置默认值 if None
            
        Returns:
            (预测的词, 概率)
            
        Raises:
            VocabularyError: 模型未训练
        """
        if self.vocab_size == 0:
            raise VocabularyError("词汇表为空，请先训练模型")
        
        strategy = strategy or self.config.sampling_strategy
        context_tuple = self._prepare_context(context)
        
        if strategy == SamplingStrategy.GREEDY:
            return self._sample_greedy(context_tuple)
        elif strategy == SamplingStrategy.WEIGHTED_RANDOM:
            return self._sample_weighted_random(context_tuple)
        
        return self._sample_greedy(context_tuple)
    
    def generate_text(
        self,
        seed: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        strategy: Optional[SamplingStrategy] = None
    ) -> str:
        """
        自动生成一段文本
        
        Args:
            seed: 种子文本（可选）
            max_length: 生成文本的最大长度，使用配置默认值 if None
            strategy: 采样策略，使用配置默认值 if None
            
        Returns:
            生成的文本字符串
            
        Raises:
            VocabularyError: 模型未训练
        """
        if self.vocab_size == 0:
            raise VocabularyError("词汇表为空，请先训练模型")
        
        max_length = max_length or self.config.max_generation_length
        strategy = strategy or self.config.sampling_strategy
        
        if seed is None:
            context = [self.config.START_TOKEN] * (self.n - 1)
        else:
            context = [self.config.START_TOKEN] * (self.n - 1) + seed
        
        generated = list(seed) if seed else []
        
        for _ in range(max_length):
            context_tuple = tuple(context[-(self.n - 1):])
            
            if strategy == SamplingStrategy.GREEDY:
                next_word, _ = self._sample_greedy(context_tuple)
            else:
                next_word, _ = self._sample_weighted_random(context_tuple)
            
            if next_word == self.config.END_TOKEN:
                break
            
            generated.append(next_word)
            context.append(next_word)
        
        return ''.join(generated)
    
    def calculate_perplexity(self, test_corpus: List[List[str]]) -> float:
        """
        计算测试集上的困惑度 (Perplexity)
        
        PP(W) = exp(-1/N * sum(log P(w_i | w_{i-n+1}...w_{i-1})))
        
        Args:
            test_corpus: 测试语料库
            
        Returns:
            困惑度值
            
        Raises:
            VocabularyError: 模型未训练
            EmptyCorpusError: 测试语料为空或无有效词
        """
        if self.vocab_size == 0:
            raise VocabularyError("词汇表为空，请先训练模型")
        
        validate_corpus(test_corpus)
        
        log_prob_sum = 0.0
        total_words = 0
        
        for sentence in test_corpus:
            padded = self._pad_sentence(sentence)
            
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                
                prob = self._get_probability(word, context)
                log_prob_sum += math.log(prob)
                total_words += 1
        
        if total_words == 0:
            raise EmptyCorpusError("测试语料中没有有效词用于计算困惑度")
        
        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取模型配置信息
        
        Returns:
            配置信息字典
        """
        return {
            'n': self.n,
            'vocab_size': self.vocab_size,
            'total_tokens': self.total_tokens,
            'smoothing_type': self.config.smoothing_type.value,
            'sampling_strategy': self.config.sampling_strategy.value,
            'random_seed': self.config.random_seed
        }
