#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 语言模型实现
支持 N 可调，包含多种平滑技术和采样策略
"""

import random
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod

from config import (
    ModelConfig, 
    GenerationConfig, 
    SmoothingMethod, 
    SamplingStrategy,
    LogConfig
)
from utils import (
    setup_logger,
    validate_corpus,
    EmptyCorpusError,
    InvalidContextError,
    InvalidInputTypeError,
    ModelNotTrainedError,
    DivisionByZeroError,
    NGramError
)


class BaseSmoothing(ABC):
    """平滑方法基类"""
    
    @abstractmethod
    def calculate_probability(
        self, 
        word: str, 
        context: Tuple[str, ...],
        ngram_counts: Dict[Tuple[str, ...], Counter],
        context_counts: Counter,
        vocab_size: int
    ) -> float:
        """计算平滑后的条件概率"""
        pass


class AddOneSmoothing(BaseSmoothing):
    """Add-1 (Laplace) 平滑"""
    
    def calculate_probability(
        self, 
        word: str, 
        context: Tuple[str, ...],
        ngram_counts: Dict[Tuple[str, ...], Counter],
        context_counts: Counter,
        vocab_size: int
    ) -> float:
        count_context_word = ngram_counts[context].get(word, 0)
        count_context = context_counts.get(context, 0)
        return (count_context_word + 1) / (count_context + vocab_size)


class AddKSmoothing(BaseSmoothing):
    """Add-k 平滑"""
    
    def __init__(self, k: float = 0.5):
        if k <= 0:
            raise ValueError(f"k 值必须大于 0，当前值: {k}")
        self.k = k
    
    def calculate_probability(
        self, 
        word: str, 
        context: Tuple[str, ...],
        ngram_counts: Dict[Tuple[str, ...], Counter],
        context_counts: Counter,
        vocab_size: int
    ) -> float:
        count_context_word = ngram_counts[context].get(word, 0)
        count_context = context_counts.get(context, 0)
        return (count_context_word + self.k) / (count_context + self.k * vocab_size)


class NoSmoothing(BaseSmoothing):
    """无平滑（最大似然估计）"""
    
    def calculate_probability(
        self, 
        word: str, 
        context: Tuple[str, ...],
        ngram_counts: Dict[Tuple[str, ...], Counter],
        context_counts: Counter,
        vocab_size: int
    ) -> float:
        count_context_word = ngram_counts[context].get(word, 0)
        count_context = context_counts.get(context, 0)
        if count_context == 0:
            return 0.0
        return count_context_word / count_context


def get_smoothing_method(config: ModelConfig) -> BaseSmoothing:
    """根据配置获取平滑方法实例"""
    if config.smoothing_method == SmoothingMethod.ADD_ONE:
        return AddOneSmoothing()
    elif config.smoothing_method == SmoothingMethod.ADD_K:
        return AddKSmoothing(k=config.smoothing_k)
    elif config.smoothing_method == SmoothingMethod.NONE:
        return NoSmoothing()
    else:
        raise ValueError(f"不支持的平滑方法: {config.smoothing_method}")


class NGramModel:
    """
    N-Gram 语言模型类
    
    支持计算条件概率 P(w_n | w_{n-(N-1)}, ..., w_{n-1})
    支持多种平滑技术和采样策略
    
    Attributes:
        config: 模型配置
        n: N-Gram 的阶数
        vocab: 词汇表
        vocab_size: 词汇表大小
        total_tokens: 总词数
        smoothing: 平滑方法实例
        logger: 日志记录器
    """
    
    def __init__(
        self, 
        config: Optional[ModelConfig] = None,
        log_config: Optional[LogConfig] = None,
        n: Optional[int] = None
    ):
        """
        初始化 N-Gram 模型
        
        Args:
            config: 模型配置对象，如果提供则忽略 n 参数
            log_config: 日志配置对象
            n: N-Gram 的阶数（向后兼容，如果 config 未提供则使用此值）
        """
        if config is not None:
            self.config = config
        else:
            self.config = ModelConfig(n=n if n is not None else 2)
        
        self.log_config = log_config or LogConfig()
        self.logger = setup_logger(self.__class__.__name__, self.log_config)
        
        self.n = self.config.n
        self.ngram_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: set = set()
        self.vocab_size: int = 0
        self.total_tokens: int = 0
        self._is_trained: bool = False
        
        self.smoothing = get_smoothing_method(self.config)
    
    def _pad_sentence(self, sentence: List[str]) -> List[str]:
        """
        为句子添加起始和结束标记
        
        Args:
            sentence: 分词后的句子列表
            
        Returns:
            添加标记后的句子
        """
        padded = ['<START>'] * (self.n - 1) + sentence + ['<END>']
        return padded
    
    def train(self, corpus: List[List[str]]) -> None:
        """
        训练 N-Gram 模型
        
        Args:
            corpus: 语料库，每个元素是一个分词后的句子列表
            
        Raises:
            InvalidInputTypeError: 输入类型错误
            EmptyCorpusError: 语料库为空
        """
        validate_corpus(corpus)
        
        self.logger.info(f"开始训练 {self.n}-Gram 模型...")
        self.logger.info(f"语料库包含 {len(corpus)} 个句子")
        
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        self.total_tokens = 0
        
        for sentence in corpus:
            padded = self._pad_sentence(sentence)
            self.total_tokens += len(sentence)
            
            for word in sentence:
                self.vocab.add(word)
            self.vocab.add('<END>')
            
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        
        self.vocab_size = len(self.vocab)
        self._is_trained = True
        
        total_ngrams = sum(len(counts) for counts in self.ngram_counts.values())
        self.logger.info(f"词汇表大小: {self.vocab_size}")
        self.logger.info(f"总词数: {self.total_tokens}")
        self.logger.info(f"不同 {self.n}-Gram 数量: {total_ngrams}")
        self.logger.info(f"不同上下文数量: {len(self.context_counts)}")
        self.logger.info("训练完成！")
    
    def _get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        计算条件概率 P(word | context)
        
        Args:
            word: 当前词
            context: 上下文 (n-1)-gram
            
        Returns:
            平滑后的条件概率
            
        Raises:
            ModelNotTrainedError: 模型未训练
        """
        if not self._is_trained:
            raise ModelNotTrainedError()
        
        return self.smoothing.calculate_probability(
            word, context, self.ngram_counts, self.context_counts, self.vocab_size
        )
    
    def _validate_context(self, context: List[str]) -> None:
        """
        校验上下文有效性
        
        Args:
            context: 上下文列表
            
        Raises:
            InvalidInputTypeError: 输入类型错误
            InvalidContextError: 上下文无效
        """
        if not isinstance(context, list):
            raise InvalidInputTypeError("List[str]", type(context).__name__)
        
        for i, word in enumerate(context):
            if not isinstance(word, str):
                raise InvalidInputTypeError(f"str (位置 {i})", type(word).__name__)
    
    def _prepare_context(self, context: List[str]) -> Tuple[str, ...]:
        """
        准备上下文元组
        
        Args:
            context: 原始上下文列表
            
        Returns:
            处理后的上下文元组
        """
        self._validate_context(context)
        
        if len(context) >= self.n - 1:
            return tuple(context[-(self.n - 1):])
        else:
            padded_context = ['<START>'] * (self.n - 1 - len(context)) + context
            return tuple(padded_context)
    
    def predict_next(self, context: List[str]) -> Tuple[str, float]:
        """
        给定前缀，预测下一个概率最大的词
        
        Args:
            context: 前缀上下文列表
            
        Returns:
            (预测的词, 概率)
            
        Raises:
            ModelNotTrainedError: 模型未训练
            InvalidInputTypeError: 输入类型错误
        """
        if not self._is_trained:
            raise ModelNotTrainedError()
        
        context_tuple = self._prepare_context(context)
        
        best_word = None
        best_prob = -1.0
        
        for word in self.vocab:
            prob = self._get_probability(word, context_tuple)
            if prob > best_prob:
                best_prob = prob
                best_word = word
        
        return best_word, best_prob
    
    def _greedy_sample(self, context_tuple: Tuple[str, ...]) -> str:
        """贪婪采样：选择概率最高的词"""
        best_word = None
        best_prob = -1.0
        
        for word in self.vocab:
            prob = self._get_probability(word, context_tuple)
            if prob > best_prob:
                best_prob = prob
                best_word = word
        
        return best_word
    
    def _weighted_random_sample(self, context_tuple: Tuple[str, ...]) -> str:
        """加权随机采样：按概率分布采样"""
        words = list(self.vocab)
        probabilities = [self._get_probability(w, context_tuple) for w in words]
        
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(words)
        
        probabilities = [p / total_prob for p in probabilities]
        return random.choices(words, weights=probabilities, k=1)[0]
    
    def _get_sampler(self, strategy: SamplingStrategy) -> Callable[[Tuple[str, ...]], str]:
        """获取采样函数"""
        if strategy == SamplingStrategy.GREEDY:
            return self._greedy_sample
        elif strategy == SamplingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_sample
        else:
            raise ValueError(f"不支持的采样策略: {strategy}")
    
    def generate_text(
        self, 
        seed: Optional[List[str]] = None, 
        max_length: int = 20,
        sampling_strategy: Optional[SamplingStrategy] = None
    ) -> str:
        """
        自动生成一段文本
        
        Args:
            seed: 种子文本（可选）
            max_length: 生成文本的最大长度
            sampling_strategy: 采样策略（贪婪/加权随机）
            
        Returns:
            生成的文本字符串
            
        Raises:
            ModelNotTrainedError: 模型未训练
            InvalidInputTypeError: 输入类型错误
        """
        if not self._is_trained:
            raise ModelNotTrainedError()
        
        if seed is not None:
            self._validate_context(seed)
        
        if sampling_strategy is None:
            sampling_strategy = SamplingStrategy.WEIGHTED_RANDOM
        
        sampler = self._get_sampler(sampling_strategy)
        
        if seed is None:
            context = ['<START>'] * (self.n - 1)
        else:
            context = ['<START>'] * (self.n - 1) + seed
        
        generated = list(seed) if seed else []
        
        for _ in range(max_length):
            context_tuple = tuple(context[-(self.n - 1):])
            next_word = sampler(context_tuple)
            
            if next_word == '<END>':
                break
            
            generated.append(next_word)
            context.append(next_word)
        
        return ' '.join(generated)
    
    def calculate_perplexity(self, test_corpus: List[List[str]]) -> float:
        """
        计算测试集上的困惑度 (Perplexity)
        
        PP(W) = exp(-1/N * sum(log P(w_i | w_{i-n+1}...w_{i-1})))
        
        Args:
            test_corpus: 测试语料库
            
        Returns:
            困惑度值
            
        Raises:
            ModelNotTrainedError: 模型未训练
            InvalidInputTypeError: 输入类型错误
            EmptyCorpusError: 语料库为空
            DivisionByZeroError: 除零错误
        """
        if not self._is_trained:
            raise ModelNotTrainedError()
        
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
                
                if prob <= 0:
                    prob = 1e-10
                
                log_prob_sum += math.log(prob)
                total_words += 1
        
        if total_words == 0:
            raise DivisionByZeroError("测试语料库中没有有效的词")
        
        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def get_probability_distribution(self, context: List[str]) -> Dict[str, float]:
        """
        获取给定上下文下所有词的概率分布
        
        Args:
            context: 上下文列表
            
        Returns:
            词到概率的映射字典
            
        Raises:
            ModelNotTrainedError: 模型未训练
        """
        if not self._is_trained:
            raise ModelNotTrainedError()
        
        context_tuple = self._prepare_context(context)
        
        distribution = {}
        for word in self.vocab:
            distribution[word] = self._get_probability(word, context_tuple)
        
        return distribution
    
    def is_trained(self) -> bool:
        """检查模型是否已训练"""
        return self._is_trained
    
    def get_vocab(self) -> set:
        """获取词汇表"""
        return self.vocab.copy()
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.vocab_size
