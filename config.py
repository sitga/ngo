#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块 - 统一管理 N-Gram 模型相关参数
"""
import os
from enum import Enum
from typing import Optional


class SmoothingType(Enum):
    """平滑方式枚举"""
    ADD_1 = "add_1"
    ADD_K = "add_k"


class SamplingStrategy(Enum):
    """采样策略枚举"""
    GREEDY = "greedy"
    WEIGHTED_RANDOM = "weighted_random"


class ModelConfig:
    """模型配置类"""
    
    def __init__(
        self,
        n: int = 2,
        random_seed: int = 42,
        train_ratio: float = 0.8,
        max_generation_length: int = 20,
        smoothing_type: SmoothingType = SmoothingType.ADD_1,
        add_k: float = 1.0,
        sampling_strategy: SamplingStrategy = SamplingStrategy.WEIGHTED_RANDOM,
        start_token: str = '<START>',
        end_token: str = '<END>',
        unknown_token: str = '<UNK>'
    ):
        self.n = n
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.max_generation_length = max_generation_length
        self.smoothing_type = smoothing_type
        self.add_k = add_k
        self.sampling_strategy = sampling_strategy
        self.START_TOKEN = start_token
        self.END_TOKEN = end_token
        self.UNKNOWN_TOKEN = unknown_token
    
    def __repr__(self) -> str:
        return (
            f"ModelConfig(n={self.n}, random_seed={self.random_seed}, "
            f"train_ratio={self.train_ratio}, "
            f"max_generation_length={self.max_generation_length}, "
            f"smoothing_type={self.smoothing_type.value}, "
            f"sampling_strategy={self.sampling_strategy.value})"
        )


class LogConfig:
    """日志配置类"""
    
    def __init__(
        self,
        log_level: str = 'INFO',
        log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_file: Optional[str] = None,
        log_to_console: bool = True
    ):
        self.log_level = log_level
        self.log_format = log_format
        self.log_file = log_file
        self.log_to_console = log_to_console


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_LOG_CONFIG = LogConfig()
