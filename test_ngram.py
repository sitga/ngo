#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 模型单元测试
测试核心功能和异常处理
"""

import unittest
import math
from typing import List

from config import (
    ModelConfig, 
    TrainingConfig, 
    GenerationConfig,
    LogConfig,
    SmoothingMethod,
    SamplingStrategy,
    Config
)
from ngram_model import (
    NGramModel, 
    AddOneSmoothing, 
    AddKSmoothing, 
    NoSmoothing
)
from utils import (
    generate_corpus,
    split_corpus,
    validate_corpus,
    setup_logger,
    EmptyCorpusError,
    InvalidInputTypeError,
    ModelNotTrainedError,
    DivisionByZeroError
)


class TestConfig(unittest.TestCase):
    """配置类测试"""
    
    def test_model_config_default(self):
        """测试模型配置默认值"""
        config = ModelConfig()
        self.assertEqual(config.n, 2)
        self.assertEqual(config.smoothing_method, SmoothingMethod.ADD_ONE)
        self.assertEqual(config.smoothing_k, 1.0)
    
    def test_model_config_invalid_n(self):
        """测试模型配置无效 n 值"""
        with self.assertRaises(ValueError):
            ModelConfig(n=0)
        with self.assertRaises(ValueError):
            ModelConfig(n=-1)
    
    def test_training_config_invalid_ratio(self):
        """测试训练配置无效比例"""
        with self.assertRaises(ValueError):
            TrainingConfig(train_ratio=0)
        with self.assertRaises(ValueError):
            TrainingConfig(train_ratio=1)
        with self.assertRaises(ValueError):
            TrainingConfig(train_ratio=-0.5)
    
    def test_generation_config_invalid_max_length(self):
        """测试生成配置无效最大长度"""
        with self.assertRaises(ValueError):
            GenerationConfig(max_length=0)
        with self.assertRaises(ValueError):
            GenerationConfig(max_length=-1)
    
    def test_log_config_invalid_level(self):
        """测试日志配置无效级别"""
        with self.assertRaises(ValueError):
            LogConfig(log_level="INVALID")
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            "model": {"n": 3, "smoothing_method": "add_k", "smoothing_k": 0.5},
            "training": {"random_seed": 123, "train_ratio": 0.75},
            "generation": {"max_length": 15, "sampling_strategy": "greedy"},
            "log": {"log_level": "DEBUG"}
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.model.n, 3)
        self.assertEqual(config.model.smoothing_method, SmoothingMethod.ADD_K)
        self.assertEqual(config.training.random_seed, 123)
        self.assertEqual(config.generation.sampling_strategy, SamplingStrategy.GREEDY)
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = Config()
        config_dict = config.to_dict()
        self.assertIn("model", config_dict)
        self.assertIn("training", config_dict)
        self.assertIn("generation", config_dict)
        self.assertIn("log", config_dict)


class TestSmoothingMethods(unittest.TestCase):
    """平滑方法测试"""
    
    def setUp(self):
        """测试前准备"""
        self.ngram_counts = {("今",): {"天": 5, "年": 2}}
        self.context_counts = {("今",): 7}
        self.vocab_size = 100
    
    def test_add_one_smoothing(self):
        """测试 Add-1 平滑"""
        smoothing = AddOneSmoothing()
        prob = smoothing.calculate_probability(
            "天", ("今",), self.ngram_counts, self.context_counts, self.vocab_size
        )
        expected = (5 + 1) / (7 + 100)
        self.assertAlmostEqual(prob, expected, places=6)
    
    def test_add_one_smoothing_unseen_word(self):
        """测试 Add-1 平滑未见词"""
        smoothing = AddOneSmoothing()
        prob = smoothing.calculate_probability(
            "未", ("今",), self.ngram_counts, self.context_counts, self.vocab_size
        )
        expected = (0 + 1) / (7 + 100)
        self.assertAlmostEqual(prob, expected, places=6)
    
    def test_add_k_smoothing(self):
        """测试 Add-k 平滑"""
        smoothing = AddKSmoothing(k=0.5)
        prob = smoothing.calculate_probability(
            "天", ("今",), self.ngram_counts, self.context_counts, self.vocab_size
        )
        expected = (5 + 0.5) / (7 + 0.5 * 100)
        self.assertAlmostEqual(prob, expected, places=6)
    
    def test_add_k_smoothing_invalid_k(self):
        """测试 Add-k 平滑无效 k 值"""
        with self.assertRaises(ValueError):
            AddKSmoothing(k=0)
        with self.assertRaises(ValueError):
            AddKSmoothing(k=-1)
    
    def test_no_smoothing(self):
        """测试无平滑"""
        smoothing = NoSmoothing()
        prob = smoothing.calculate_probability(
            "天", ("今",), self.ngram_counts, self.context_counts, self.vocab_size
        )
        expected = 5 / 7
        self.assertAlmostEqual(prob, expected, places=6)
    
    def test_no_smoothing_unseen_word(self):
        """测试无平滑未见词"""
        smoothing = NoSmoothing()
        prob = smoothing.calculate_probability(
            "未", ("今",), self.ngram_counts, self.context_counts, self.vocab_size
        )
        self.assertEqual(prob, 0.0)


class TestNGramModel(unittest.TestCase):
    """NGramModel 类测试"""
    
    def setUp(self):
        """测试前准备"""
        self.log_config = LogConfig(log_level="WARNING")
        self.corpus: List[List[str]] = [
            ["今", "天", "天", "气", "很", "好"],
            ["明", "天", "天", "气", "也", "不", "错"],
            ["今", "天", "我", "很", "开", "心"],
            ["明", "天", "我", "要", "出", "门"],
        ]
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = NGramModel(n=2, log_config=self.log_config)
        self.assertEqual(model.n, 2)
        self.assertFalse(model.is_trained())
    
    def test_model_initialization_with_config(self):
        """测试使用配置对象初始化模型"""
        config = ModelConfig(n=3)
        model = NGramModel(config=config, log_config=self.log_config)
        self.assertEqual(model.n, 3)
    
    def test_train(self):
        """测试模型训练"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        self.assertTrue(model.is_trained())
        self.assertGreater(model.get_vocab_size(), 0)
    
    def test_train_empty_corpus(self):
        """测试空语料库训练"""
        model = NGramModel(n=2, log_config=self.log_config)
        with self.assertRaises(EmptyCorpusError):
            model.train([])
    
    def test_train_invalid_corpus_type(self):
        """测试无效语料库类型"""
        model = NGramModel(n=2, log_config=self.log_config)
        with self.assertRaises(InvalidInputTypeError):
            model.train("not a list")
        with self.assertRaises(InvalidInputTypeError):
            model.train([["今", "天"], "not a list"])
    
    def test_predict_next_before_train(self):
        """测试训练前预测"""
        model = NGramModel(n=2, log_config=self.log_config)
        with self.assertRaises(ModelNotTrainedError):
            model.predict_next(["今"])
    
    def test_predict_next(self):
        """测试预测下一个词"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        word, prob = model.predict_next(["今"])
        self.assertIn(word, model.get_vocab())
        self.assertGreater(prob, 0)
    
    def test_predict_next_invalid_context(self):
        """测试无效上下文预测"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        with self.assertRaises(InvalidInputTypeError):
            model.predict_next("not a list")
        with self.assertRaises(InvalidInputTypeError):
            model.predict_next([1, 2, 3])
    
    def test_generate_text(self):
        """测试文本生成"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        text = model.generate_text(max_length=10)
        self.assertIsInstance(text, str)
    
    def test_generate_text_with_seed(self):
        """测试带种子的文本生成"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        text = model.generate_text(seed=["今", "天"], max_length=10)
        self.assertTrue(text.startswith("今 天"))
    
    def test_generate_text_greedy(self):
        """测试贪婪采样生成"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        text1 = model.generate_text(
            seed=["今"], 
            max_length=5, 
            sampling_strategy=SamplingStrategy.GREEDY
        )
        text2 = model.generate_text(
            seed=["今"], 
            max_length=5, 
            sampling_strategy=SamplingStrategy.GREEDY
        )
        self.assertEqual(text1, text2)
    
    def test_generate_text_before_train(self):
        """测试训练前生成文本"""
        model = NGramModel(n=2, log_config=self.log_config)
        with self.assertRaises(ModelNotTrainedError):
            model.generate_text()
    
    def test_calculate_perplexity(self):
        """测试困惑度计算"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        test_corpus = [["今", "天", "很", "好"]]
        perplexity = model.calculate_perplexity(test_corpus)
        self.assertGreater(perplexity, 0)
        self.assertTrue(math.isfinite(perplexity))
    
    def test_calculate_perplexity_before_train(self):
        """测试训练前计算困惑度"""
        model = NGramModel(n=2, log_config=self.log_config)
        with self.assertRaises(ModelNotTrainedError):
            model.calculate_perplexity([["今", "天"]])
    
    def test_calculate_perplexity_empty_corpus(self):
        """测试空语料库困惑度"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        with self.assertRaises(EmptyCorpusError):
            model.calculate_perplexity([])
    
    def test_get_probability_distribution(self):
        """测试获取概率分布"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        distribution = model.get_probability_distribution(["今"])
        self.assertIsInstance(distribution, dict)
        total_prob = sum(distribution.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        model = NGramModel(n=2, log_config=self.log_config)
        model.train(self.corpus)
        self.assertEqual(model.n, 2)
        
        word, prob = model.predict_next(["今"])
        self.assertIsNotNone(word)


class TestUtils(unittest.TestCase):
    """工具函数测试"""
    
    def test_generate_corpus(self):
        """测试语料生成"""
        corpus = generate_corpus()
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        for sentence in corpus:
            self.assertIsInstance(sentence, list)
            for word in sentence:
                self.assertIsInstance(word, str)
    
    def test_split_corpus(self):
        """测试语料划分"""
        corpus = generate_corpus()
        train, test = split_corpus(corpus, train_ratio=0.8, random_seed=42)
        self.assertEqual(len(train) + len(test), len(corpus))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
    
    def test_split_corpus_reproducible(self):
        """测试语料划分可重复性"""
        corpus = generate_corpus()
        train1, test1 = split_corpus(corpus, train_ratio=0.8, random_seed=42)
        train2, test2 = split_corpus(corpus, train_ratio=0.8, random_seed=42)
        self.assertEqual(train1, train2)
        self.assertEqual(test1, test2)
    
    def test_validate_corpus_valid(self):
        """测试有效语料校验"""
        corpus = [["今", "天"], ["明", "天"]]
        validate_corpus(corpus)
    
    def test_validate_corpus_empty(self):
        """测试空语料校验"""
        with self.assertRaises(EmptyCorpusError):
            validate_corpus([])
    
    def test_validate_corpus_invalid_type(self):
        """测试无效类型语料校验"""
        with self.assertRaises(InvalidInputTypeError):
            validate_corpus("not a list")
        with self.assertRaises(InvalidInputTypeError):
            validate_corpus([["今", "天"], "not a list"])
        with self.assertRaises(InvalidInputTypeError):
            validate_corpus([[1, 2, 3]])
    
    def test_setup_logger(self):
        """测试日志配置"""
        log_config = LogConfig(log_level="DEBUG")
        logger = setup_logger("test_logger", log_config)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, 10)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        log_config = LogConfig(log_level="WARNING")
        
        corpus = generate_corpus()
        train_corpus, test_corpus = split_corpus(corpus, train_ratio=0.8, random_seed=42)
        
        model = NGramModel(n=2, log_config=log_config)
        model.train(train_corpus)
        
        word, prob = model.predict_next(["今", "天"])
        self.assertIsNotNone(word)
        
        text = model.generate_text(max_length=10)
        self.assertIsInstance(text, str)
        
        perplexity = model.calculate_perplexity(test_corpus)
        self.assertGreater(perplexity, 0)
    
    def test_different_n_values(self):
        """测试不同 N 值"""
        log_config = LogConfig(log_level="WARNING")
        corpus = generate_corpus()
        train_corpus, test_corpus = split_corpus(corpus, train_ratio=0.8, random_seed=42)
        
        for n in [2, 3, 4]:
            model = NGramModel(n=n, log_config=log_config)
            model.train(train_corpus)
            perplexity = model.calculate_perplexity(test_corpus)
            self.assertGreater(perplexity, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
