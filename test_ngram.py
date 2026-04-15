#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 模型单元测试
"""
import unittest
import logging

from config import ModelConfig, SmoothingType, SamplingStrategy
from ngram_model import NGramModel
from utils import (
    generate_corpus,
    split_corpus,
    validate_corpus,
    EmptyCorpusError,
    InvalidContextError,
    InvalidInputError,
    VocabularyError
)

logging.basicConfig(level=logging.ERROR)


class TestNGramModel(unittest.TestCase):
    """N-Gram 模型核心功能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.simple_corpus = [
            list("我喜欢吃苹果"),
            list("我喜欢吃香蕉"),
            list("今天天气真好"),
            list("明天天气不好")
        ]
        self.config = ModelConfig(n=2, random_seed=42)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = NGramModel(self.config)
        self.assertEqual(model.n, 2)
        self.assertEqual(model.vocab_size, 0)
        self.assertEqual(model.total_tokens, 0)
        
        with self.assertRaises(InvalidInputError):
            bad_config = ModelConfig(n=1)
            NGramModel(bad_config)
    
    def test_train_basic(self):
        """测试基本训练功能"""
        model = NGramModel(self.config)
        model.train(self.simple_corpus)
        
        self.assertGreater(model.vocab_size, 0)
        self.assertGreater(model.total_tokens, 0)
        self.assertGreater(len(model.ngram_counts), 0)
    
    def test_train_empty_corpus(self):
        """测试空语料库训练"""
        model = NGramModel(self.config)
        with self.assertRaises(EmptyCorpusError):
            model.train([])
    
    def test_predict_next_before_train(self):
        """测试未训练模型预测"""
        model = NGramModel(self.config)
        with self.assertRaises(VocabularyError):
            model.predict_next(["今", "天"])
    
    def test_predict_next_basic(self):
        """测试基本预测功能"""
        model = NGramModel(self.config)
        model.train(self.simple_corpus)
        
        next_word, prob = model.predict_next(["我", "喜"])
        self.assertIsInstance(next_word, str)
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_predict_next_strategies(self):
        """测试不同采样策略"""
        model = NGramModel(self.config)
        model.train(self.simple_corpus)
        
        greedy_word, _ = model.predict_next(
            ["天", "气"], strategy=SamplingStrategy.GREEDY
        )
        random_word, _ = model.predict_next(
            ["天", "气"], strategy=SamplingStrategy.WEIGHTED_RANDOM
        )
        
        self.assertIsInstance(greedy_word, str)
        self.assertIsInstance(random_word, str)
    
    def test_generate_text_before_train(self):
        """测试未训练模型生成文本"""
        model = NGramModel(self.config)
        with self.assertRaises(VocabularyError):
            model.generate_text()
    
    def test_generate_text_basic(self):
        """测试基本文本生成"""
        model = NGramModel(self.config)
        model.train(self.simple_corpus)
        
        text1 = model.generate_text()
        self.assertIsInstance(text1, str)
        
        text2 = model.generate_text(seed=["今", "天"], max_length=5)
        self.assertIsInstance(text2, str)
        self.assertLessEqual(len(text2), 7)
    
    def test_calculate_perplexity_before_train(self):
        """测试未训练模型计算困惑度"""
        model = NGramModel(self.config)
        with self.assertRaises(VocabularyError):
            model.calculate_perplexity(self.simple_corpus)
    
    def test_calculate_perplexity_basic(self):
        """测试基本困惑度计算"""
        model = NGramModel(self.config)
        model.train(self.simple_corpus)
        
        test_corpus = [list("今天天气好")]
        perplexity = model.calculate_perplexity(test_corpus)
        
        self.assertGreater(perplexity, 0)
    
    def test_different_n_values(self):
        """测试不同 N 值的模型"""
        for n in [2, 3, 4]:
            config = ModelConfig(n=n, random_seed=42)
            model = NGramModel(config)
            model.train(self.simple_corpus)
            self.assertEqual(model.n, n)
            
            text = model.generate_text(max_length=5)
            self.assertIsInstance(text, str)
    
    def test_add_k_smoothing(self):
        """测试 Add-k 平滑"""
        config = ModelConfig(
            n=2,
            smoothing_type=SmoothingType.ADD_K,
            add_k=0.5,
            random_seed=42
        )
        model = NGramModel(config)
        model.train(self.simple_corpus)
        
        prob = model._get_probability("气", ('今',))
        self.assertGreater(prob, 0)
    
    def test_get_config(self):
        """测试获取配置信息"""
        model = NGramModel(self.config)
        model.train(self.simple_corpus)
        
        config_info = model.get_config()
        self.assertIn('n', config_info)
        self.assertIn('vocab_size', config_info)
        self.assertIn('smoothing_type', config_info)


class TestUtils(unittest.TestCase):
    """工具函数测试"""
    
    def test_generate_corpus(self):
        """测试语料生成"""
        corpus = generate_corpus()
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        self.assertIsInstance(corpus[0], list)
    
    def test_split_corpus(self):
        """测试语料划分"""
        corpus = generate_corpus()
        train, test = split_corpus(corpus, train_ratio=0.8, random_seed=42)
        
        self.assertEqual(len(train) + len(test), len(corpus))
        self.assertGreater(len(train), len(test))
    
    def test_split_corpus_invalid_ratio(self):
        """测试无效划分比例"""
        corpus = generate_corpus()
        with self.assertRaises(InvalidInputError):
            split_corpus(corpus, train_ratio=1.5)
        
        with self.assertRaises(InvalidInputError):
            split_corpus(corpus, train_ratio=0)
    
    def test_split_corpus_empty(self):
        """测试空语料划分"""
        with self.assertRaises(InvalidInputError):
            split_corpus([])
    
    def test_validate_corpus(self):
        """测试语料验证"""
        valid_corpus = [["我", "喜", "欢"]]
        validate_corpus(valid_corpus)
        
        with self.assertRaises(EmptyCorpusError):
            validate_corpus([])
        
        with self.assertRaises(InvalidInputError):
            validate_corpus("not a list")


def run_all_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestNGramModel))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_all_tests()
