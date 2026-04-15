#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 语言模型单元测试

测试核心功能：训练、概率计算、预测、文本生成、困惑度计算
"""

import unittest
import math

from config import (
    ModelConfig,
    GenerationConfig,
    DataConfig,
    LoggingConfig,
    SmoothingMethod,
    SamplingStrategy
)
from utils import (
    generate_corpus,
    split_corpus,
    setup_logging,
    validate_corpus,
    tokenize_text,
    detokenize_text,
    get_corpus_stats
)
from ngram_model import (
    NGramModel,
    NGramModelError,
    EmptyCorpusError,
    UntrainedModelError,
    InvalidContextError
)


class TestNGramModel(unittest.TestCase):
    """N-Gram 模型测试类"""

    def setUp(self):
        """测试前准备"""
        self.logging_config = LoggingConfig(level="WARNING")
        self.logger = setup_logging(self.logging_config)

        # 创建简单的测试语料库
        self.test_corpus = [
            list("今天天气很好"),
            list("明天会下雨"),
            list("天气真不错"),
            list("今天出去玩"),
            list("明天要上班")
        ]

        self.model_config = ModelConfig(n=2)
        self.generation_config = GenerationConfig(max_length=10)
        self.data_config = DataConfig()

    def test_model_initialization(self):
        """测试模型初始化"""
        model = NGramModel(
            model_config=self.model_config,
            generation_config=self.generation_config,
            data_config=self.data_config,
            logger=self.logger
        )

        self.assertEqual(model.n, 2)
        self.assertEqual(model.vocab_size, 0)
        self.assertEqual(model.total_tokens, 0)
        self.assertFalse(model._is_trained)

    def test_train_basic(self):
        """测试基本训练功能"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        self.assertTrue(model._is_trained)
        self.assertGreater(model.vocab_size, 0)
        self.assertGreater(model.total_tokens, 0)
        self.assertGreater(len(model.ngram_counts), 0)

    def test_train_empty_corpus(self):
        """测试空语料库异常"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)

        with self.assertRaises(EmptyCorpusError):
            model.train([])

    def test_train_invalid_corpus_type(self):
        """测试无效语料库类型异常"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)

        with self.assertRaises(TypeError):
            model.train("invalid corpus")

    def test_untrained_model_error(self):
        """测试未训练模型异常"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)

        with self.assertRaises(UntrainedModelError):
            model.predict_next(["今", "天"])

        with self.assertRaises(UntrainedModelError):
            model.generate_text()

        with self.assertRaises(UntrainedModelError):
            model.calculate_perplexity(self.test_corpus)

    def test_get_probability(self):
        """测试概率计算"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        # 测试已知 N-Gram 的概率
        prob = model._get_probability("天", ("今",))
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1)

        # 测试未知 N-Gram 的概率（应该使用平滑）
        prob_unknown = model._get_probability("未", ("知",))
        self.assertGreater(prob_unknown, 0)
        self.assertLessEqual(prob_unknown, 1)

    def test_predict_next(self):
        """测试下一个词预测"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        next_word, prob = model.predict_next(["今", "天"])

        self.assertIsInstance(next_word, str)
        self.assertIsInstance(prob, float)
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1)

    def test_predict_next_short_context(self):
        """测试短上下文预测"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        # 上下文长度不足 n-1
        next_word, prob = model.predict_next(["今"])

        self.assertIsInstance(next_word, str)
        self.assertGreater(prob, 0)

    def test_predict_next_invalid_context(self):
        """测试无效上下文异常"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        with self.assertRaises(InvalidContextError):
            model.predict_next([123, 456])  # 非字符串元素

    def test_generate_text(self):
        """测试文本生成"""
        model = NGramModel(
            model_config=self.model_config,
            generation_config=self.generation_config,
            logger=self.logger
        )
        model.train(self.test_corpus)

        generated = model.generate_text(max_length=5)

        self.assertIsInstance(generated, str)
        self.assertLessEqual(len(generated), 5)

    def test_generate_text_with_seed(self):
        """测试带种子的文本生成"""
        model = NGramModel(
            model_config=self.model_config,
            generation_config=self.generation_config,
            logger=self.logger
        )
        model.train(self.test_corpus)

        seed = ["今", "天"]
        generated = model.generate_text(seed=seed, max_length=5)

        self.assertIsInstance(generated, str)
        self.assertTrue(generated.startswith("今天") or len(generated) >= 2)

    def test_calculate_perplexity(self):
        """测试困惑度计算"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        # 使用训练集作为测试集
        perplexity = model.calculate_perplexity(self.test_corpus)

        self.assertIsInstance(perplexity, float)
        self.assertGreater(perplexity, 0)

    def test_calculate_perplexity_empty_corpus(self):
        """测试空测试集异常"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        with self.assertRaises(EmptyCorpusError):
            model.calculate_perplexity([])

    def test_get_model_info(self):
        """测试获取模型信息"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        info = model.get_model_info()

        self.assertIn("n", info)
        self.assertIn("is_trained", info)
        self.assertIn("vocab_size", info)
        self.assertIn("total_tokens", info)
        self.assertIn("smoothing_method", info)

        self.assertEqual(info["n"], 2)
        self.assertTrue(info["is_trained"])

    def test_get_ngram_probability(self):
        """测试获取 N-Gram 概率"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        prob = model.get_ngram_probability(("今", "天"))

        self.assertIsInstance(prob, float)
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1)

    def test_get_ngram_probability_invalid_length(self):
        """测试无效长度 N-Gram 异常"""
        model = NGramModel(model_config=self.model_config, logger=self.logger)
        model.train(self.test_corpus)

        with self.assertRaises(ValueError):
            model.get_ngram_probability(("今",))  # 长度应为 2

    def test_different_n_values(self):
        """测试不同 N 值"""
        for n in [1, 2, 3]:
            config = ModelConfig(n=n)
            model = NGramModel(model_config=config, logger=self.logger)
            model.train(self.test_corpus)

            self.assertEqual(model.n, n)
            self.assertTrue(model._is_trained)

            # 测试预测
            context = ["今"] * (n - 1) if n > 1 else []
            next_word, prob = model.predict_next(context)
            self.assertIsInstance(next_word, str)

    def test_different_smoothing_methods(self):
        """测试不同平滑方法"""
        methods = [
            SmoothingMethod.ADD_ONE,
            SmoothingMethod.ADD_K,
        ]

        for method in methods:
            if method == SmoothingMethod.ADD_K:
                config = ModelConfig(n=2, smoothing=method, smoothing_k=0.5)
            else:
                config = ModelConfig(n=2, smoothing=method)

            model = NGramModel(model_config=config, logger=self.logger)
            model.train(self.test_corpus)

            perplexity = model.calculate_perplexity(self.test_corpus)
            self.assertGreater(perplexity, 0)

    def test_sampling_strategies(self):
        """测试不同采样策略"""
        strategies = [
            SamplingStrategy.WEIGHTED_RANDOM,
            SamplingStrategy.GREEDY,
        ]

        for strategy in strategies:
            gen_config = GenerationConfig(sampling_strategy=strategy, max_length=5)
            model = NGramModel(
                model_config=self.model_config,
                generation_config=gen_config,
                logger=self.logger
            )
            model.train(self.test_corpus)

            generated = model.generate_text(max_length=5)
            self.assertIsInstance(generated, str)


class TestUtils(unittest.TestCase):
    """工具函数测试类"""

    def test_validate_corpus_valid(self):
        """测试有效语料库验证"""
        corpus = [
            list("今天天气很好"),
            list("明天会下雨")
        ]
        # 不应抛出异常
        validate_corpus(corpus)

    def test_validate_corpus_empty(self):
        """测试空语料库验证"""
        with self.assertRaises(ValueError):
            validate_corpus([])

    def test_validate_corpus_invalid_type(self):
        """测试无效类型语料库验证"""
        with self.assertRaises(TypeError):
            validate_corpus("invalid")

    def test_tokenize_text(self):
        """测试文本分词"""
        text = "今天天气很好"
        tokens = tokenize_text(text)

        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), len(text))
        self.assertEqual(tokens, list(text))

    def test_tokenize_text_invalid_type(self):
        """测试无效类型分词"""
        with self.assertRaises(TypeError):
            tokenize_text(123)

    def test_tokenize_text_empty(self):
        """测试空文本分词"""
        with self.assertRaises(ValueError):
            tokenize_text("")

    def test_detokenize_text(self):
        """测试文本还原"""
        tokens = ["今", "天", "天", "气", "很", "好"]
        text = detokenize_text(tokens)

        self.assertIsInstance(text, str)
        self.assertEqual(text, "今天天气很好")

    def test_get_corpus_stats(self):
        """测试语料库统计"""
        corpus = [
            list("今天天气很好"),
            list("明天会下雨")
        ]
        stats = get_corpus_stats(corpus)

        self.assertIn("total_sentences", stats)
        self.assertIn("total_words", stats)
        self.assertIn("avg_sentence_length", stats)
        self.assertIn("vocab_size", stats)

        self.assertEqual(stats["total_sentences"], 2)

    def test_generate_corpus(self):
        """测试语料库生成"""
        corpus = generate_corpus()

        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)

        for sentence in corpus:
            self.assertIsInstance(sentence, list)

    def test_split_corpus(self):
        """测试语料库划分"""
        corpus = [
            list("今天天气很好"),
            list("明天会下雨"),
            list("天气真不错"),
            list("今天出去玩"),
            list("明天要上班"),
            list("后天去旅行"),
            list("昨天下雨了"),
            list("前天天气好"),
            list("每天都很忙"),
            list("年年有今日")
        ]

        train, test = split_corpus(corpus)

        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(train) + len(test), len(corpus))


class TestConfig(unittest.TestCase):
    """配置类测试"""

    def test_model_config_validation(self):
        """测试模型配置验证"""
        # 有效配置
        config = ModelConfig(n=2)
        self.assertEqual(config.n, 2)

        # 无效 n 值
        with self.assertRaises(ValueError):
            ModelConfig(n=0)

        with self.assertRaises(ValueError):
            ModelConfig(n=-1)

    def test_generation_config_validation(self):
        """测试生成配置验证"""
        # 有效配置
        config = GenerationConfig(max_length=20)
        self.assertEqual(config.max_length, 20)

        # 无效 max_length
        with self.assertRaises(ValueError):
            GenerationConfig(max_length=0)

        with self.assertRaises(ValueError):
            GenerationConfig(max_length=-1)

    def test_data_config_validation(self):
        """测试数据配置验证"""
        # 有效配置
        config = DataConfig(train_ratio=0.8)
        self.assertEqual(config.train_ratio, 0.8)

        # 无效 train_ratio
        with self.assertRaises(ValueError):
            DataConfig(train_ratio=0)

        with self.assertRaises(ValueError):
            DataConfig(train_ratio=1.0)

        with self.assertRaises(ValueError):
            DataConfig(train_ratio=-0.1)

    def test_logging_config_validation(self):
        """测试日志配置验证"""
        # 有效配置
        config = LoggingConfig(level="INFO")
        self.assertEqual(config.level, "INFO")

        # 无效 level
        with self.assertRaises(ValueError):
            LoggingConfig(level="INVALID")


class TestIntegration(unittest.TestCase):
    """集成测试类"""

    def test_full_pipeline(self):
        """测试完整流程"""
        # 1. 生成语料库
        corpus = generate_corpus()

        # 2. 划分数据集
        train_corpus, test_corpus = split_corpus(corpus)

        # 3. 创建模型
        model_config = ModelConfig(n=2, smoothing=SmoothingMethod.ADD_ONE)
        generation_config = GenerationConfig(max_length=15)
        logging_config = LoggingConfig(level="WARNING")
        logger = setup_logging(logging_config)

        model = NGramModel(
            model_config=model_config,
            generation_config=generation_config,
            logger=logger
        )

        # 4. 训练模型
        model.train(train_corpus)

        # 5. 预测下一个词
        next_word, prob = model.predict_next(["今", "天"])
        self.assertIsInstance(next_word, str)
        self.assertGreater(prob, 0)

        # 6. 生成文本
        generated = model.generate_text(max_length=10)
        self.assertIsInstance(generated, str)

        # 7. 计算困惑度
        perplexity = model.calculate_perplexity(test_corpus)
        self.assertGreater(perplexity, 0)

        # 8. 获取模型信息
        info = model.get_model_info()
        self.assertTrue(info["is_trained"])


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestNGramModel))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
