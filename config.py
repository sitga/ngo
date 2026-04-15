#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 模型配置模块
统一管理模型参数、路径、随机种子等配置项
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SmoothingMethod(Enum):
    """平滑方法枚举"""
    ADD_ONE = "add_one"
    ADD_K = "add_k"
    NONE = "none"


class SamplingStrategy(Enum):
    """采样策略枚举"""
    GREEDY = "greedy"
    WEIGHTED_RANDOM = "weighted_random"


@dataclass
class ModelConfig:
    """模型配置类"""
    n: int = 2
    smoothing_method: SmoothingMethod = SmoothingMethod.ADD_ONE
    smoothing_k: float = 1.0
    
    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n 值必须 >= 1，当前值: {self.n}")
        if isinstance(self.smoothing_method, str):
            self.smoothing_method = SmoothingMethod(self.smoothing_method)


@dataclass
class TrainingConfig:
    """训练配置类"""
    random_seed: int = 42
    train_ratio: float = 0.8
    
    def __post_init__(self):
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio 必须在 (0, 1) 范围内，当前值: {self.train_ratio}")


@dataclass
class GenerationConfig:
    """文本生成配置类"""
    max_length: int = 20
    sampling_strategy: SamplingStrategy = SamplingStrategy.WEIGHTED_RANDOM
    
    def __post_init__(self):
        if self.max_length < 1:
            raise ValueError(f"max_length 必须 >= 1，当前值: {self.max_length}")
        if isinstance(self.sampling_strategy, str):
            self.sampling_strategy = SamplingStrategy(self.sampling_strategy)


@dataclass
class LogConfig:
    """日志配置类"""
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level 必须是 {valid_levels} 之一，当前值: {self.log_level}")
        self.log_level = self.log_level.upper()


@dataclass
class Config:
    """总配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """从字典创建配置对象"""
        model_cfg = config_dict.get("model", {})
        training_cfg = config_dict.get("training", {})
        generation_cfg = config_dict.get("generation", {})
        log_cfg = config_dict.get("log", {})
        
        return cls(
            model=ModelConfig(**model_cfg),
            training=TrainingConfig(**training_cfg),
            generation=GenerationConfig(**generation_cfg),
            log=LogConfig(**log_cfg),
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model": {
                "n": self.model.n,
                "smoothing_method": self.model.smoothing_method.value,
                "smoothing_k": self.model.smoothing_k,
            },
            "training": {
                "random_seed": self.training.random_seed,
                "train_ratio": self.training.train_ratio,
            },
            "generation": {
                "max_length": self.generation.max_length,
                "sampling_strategy": self.generation.sampling_strategy.value,
            },
            "log": {
                "log_level": self.log.log_level,
                "log_to_file": self.log.log_to_file,
                "log_file_path": self.log.log_file_path,
                "log_format": self.log.log_format,
            },
        }


DEFAULT_CONFIG = Config()
