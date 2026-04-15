#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Gram 语言模型实现
支持 N 可调，包含 Add-1 (Laplace) 平滑技术
"""

import random
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class NGramModel:
    """
    N-Gram 语言模型类
    
    支持计算条件概率 P(w_n | w_{n-(N-1)}, ..., w_{n-1})
    使用 Add-1 (Laplace) 平滑处理未见的 N-Gram 序列
    """
    
    def __init__(self, n: int = 2):
        """
        初始化 N-Gram 模型
        
        Args:
            n: N-Gram 的阶数，默认为 2 (bigram)
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)  # n-gram 计数
        self.context_counts = Counter()  # (n-1)-gram 上下文计数
        self.vocab = set()  # 词汇表
        self.vocab_size = 0  # 词汇表大小
        self.total_tokens = 0  # 总词数
        
    def _pad_sentence(self, sentence: List[str]) -> List[str]:
        """
        为句子添加起始和结束标记
        
        Args:
            sentence: 分词后的句子列表
            
        Returns:
            添加标记后的句子
        """
        # 添加 (n-1) 个 <START> 标记和 1 个 <END> 标记
        padded = ['<START>'] * (self.n - 1) + sentence + ['<END>']
        return padded
    
    def train(self, corpus: List[List[str]]) -> None:
        """
        训练 N-Gram 模型
        
        Args:
            corpus: 语料库，每个元素是一个分词后的句子列表
        """
        print(f"开始训练 {self.n}-Gram 模型...")
        print(f"语料库包含 {len(corpus)} 个句子")
        
        # 统计词频和 N-Gram
        for sentence in corpus:
            padded = self._pad_sentence(sentence)
            self.total_tokens += len(sentence)
            
            # 更新词汇表
            for word in sentence:
                self.vocab.add(word)
            self.vocab.add('<END>')
            
            # 统计 N-Gram
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]  # (n-1)-gram 上下文
                word = ngram[-1]  # 当前词
                
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
        
        self.vocab_size = len(self.vocab)
        
        # 打印训练统计信息
        total_ngrams = sum(len(counts) for counts in self.ngram_counts.values())
        print(f"词汇表大小: {self.vocab_size}")
        print(f"总词数: {self.total_tokens}")
        print(f"不同 {self.n}-Gram 数量: {total_ngrams}")
        print(f"不同上下文数量: {len(self.context_counts)}")
        print("训练完成！\n")
    
    def _get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        计算条件概率 P(word | context)，使用 Add-1 平滑
        
        Args:
            word: 当前词
            context: 上下文 (n-1)-gram
            
        Returns:
            平滑后的条件概率
        """
        # Add-1 (Laplace) 平滑
        # P(w_n | w_{n-(N-1)}, ..., w_{n-1}) = (count(context, w_n) + 1) / (count(context) + |V|)
        
        count_context_word = self.ngram_counts[context].get(word, 0)
        count_context = self.context_counts.get(context, 0)
        
        # Add-1 平滑公式
        probability = (count_context_word + 1) / (count_context + self.vocab_size)
        
        return probability
    
    def predict_next(self, context: List[str]) -> Tuple[str, float]:
        """
        给定前缀，预测下一个概率最大的词
        
        Args:
            context: 前缀上下文列表
            
        Returns:
            (预测的词, 概率)
        """
        # 确保上下文长度为 n-1
        if len(context) >= self.n - 1:
            context_tuple = tuple(context[-(self.n - 1):])
        else:
            # 如果上下文不足，用 <START> 填充
            padded_context = ['<START>'] * (self.n - 1 - len(context)) + context
            context_tuple = tuple(padded_context)
        
        # 计算所有可能词的概率
        best_word = None
        best_prob = -1
        
        for word in self.vocab:
            prob = self._get_probability(word, context_tuple)
            if prob > best_prob:
                best_prob = prob
                best_word = word
        
        return best_word, best_prob
    
    def generate_text(self, seed: Optional[List[str]] = None, 
                      max_length: int = 20) -> str:
        """
        自动生成一段文本
        
        Args:
            seed: 种子文本（可选）
            max_length: 生成文本的最大长度
            
        Returns:
            生成的文本字符串
        """
        if seed is None:
            # 从 <START> 标记开始
            context = ['<START>'] * (self.n - 1)
        else:
            context = ['<START>'] * (self.n - 1) + seed
        
        generated = list(seed) if seed else []
        
        for _ in range(max_length):
            context_tuple = tuple(context[-(self.n - 1):])
            
            # 采样下一个词（按概率分布）
            words = list(self.vocab)
            probabilities = [self._get_probability(w, context_tuple) for w in words]
            
            # 归一化概率
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]
            
            # 按概率采样
            next_word = random.choices(words, weights=probabilities, k=1)[0]
            
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
        """
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
        
        # 计算困惑度
        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity


def generate_corpus() -> List[List[str]]:
    """
    生成模拟语料库（关于天气和日常对话）
    
    Returns:
        分词后的语料库列表
    """
    sentences = [
        # 天气相关
        "今天天气真好",
        "阳光明媚适合外出",
        "天空湛蓝没有云朵",
        "气温适宜非常舒服",
        "微风拂面感觉很棒",
        "天气预报说会下雨",
        "记得带伞出门",
        "雨后空气清新",
        "彩虹出现在天边",
        "乌云密布快要下雨了",
        "雷声轰隆大雨倾盆",
        "雪花飘落冬天来了",
        "寒风刺骨需要保暖",
        "霜冻覆盖了草地",
        "雾气弥漫能见度低",
        "台风来袭注意安全",
        "冰雹砸落损坏庄稼",
        "干旱持续需要灌溉",
        "洪水泛滥紧急疏散",
        "地震突发保持冷静",
        
        # 日常对话
        "早上好你吃了吗",
        "我刚吃完早餐",
        "今天工作很忙",
        "会议持续到下午",
        "午饭吃什么好呢",
        "附近有新餐厅",
        "一起去尝尝吧",
        "这部电影很好看",
        "周末有什么安排",
        "打算去公园散步",
        "晚上一起看电影",
        "明天记得带文件",
        "谢谢你的帮助",
        "不客气应该的",
        "最近身体好吗",
        "一切都很顺利",
        "祝你生日快乐",
        "收到你的礼物了",
        "非常喜欢谢谢你",
        "好久不见甚是想念",
        
        # 简单故事
        "从前有座山",
        "山里有座庙",
        "庙里有个老和尚",
        "正在给小和尚讲故事",
        "故事的内容很有趣",
        "小和尚听得入迷",
        "夜幕降临星星出现",
        "月亮高挂在天空",
        "小河静静流淌着",
        "鱼儿在水中游动",
        "鸟儿归巢休息了",
        "花儿散发着香气",
        "小草随风摇摆",
        "大树提供阴凉",
        "孩子们在玩耍",
        "笑声回荡在田野",
        "农夫辛勤地耕作",
        "庄稼长得很好",
        "秋天收获的季节",
        "果实累累挂满枝头",
        "冬天雪花纷飞",
        "大地披上银装",
        "春天万物复苏",
        "花朵竞相开放",
        "夏天绿树成荫",
        "蝉鸣声声入耳",
        "小猫在晒太阳",
        "小狗追着蝴蝶跑",
        "兔子蹦蹦跳跳",
        "鸟儿在枝头歌唱",
        "蜜蜂忙着采蜜",
        "蝴蝶翩翩起舞",
        "蜻蜓点水产卵",
        "青蛙呱呱叫唤",
        "蚂蚁搬运食物",
        "蜘蛛织网捕虫",
        "蜗牛慢慢爬行",
        "乌龟悠闲散步",
        "大象体型庞大",
        "老虎凶猛威武",
        "猴子活泼好动",
        "熊猫憨态可掬",
        "长颈鹿脖子很长",
        "斑马身上有条纹",
        "狮子是草原之王",
        "孔雀开屏美丽",
        "天鹅优雅高贵",
        "企鹅摇摇摆摆",
        "海豚聪明伶俐",
        "鲸鱼体型巨大",
        "鲨鱼海洋霸主",
        "金鱼色彩鲜艳",
        "乌龟长寿吉祥",
        "燕子春天归来",
        "大雁南飞过冬",
        "喜鹊报喜到来",
        "乌鸦聪明机智",
        "鹦鹉学舌有趣",
        "鸽子象征和平",
        "老鹰翱翔天空",
        "猫头鹰夜间活动",
        "啄木鸟树木医生",
        "黄莺歌声婉转",
        "布谷鸟催促播种",
        "公鸡清晨打鸣",
        "母鸡下蛋孵小鸡",
        "鸭子水中嬉戏",
        "鹅伸长脖子叫",
        "马儿奔跑迅速",
        "牛羊吃草悠闲",
        "猪在泥里打滚",
        "狗看家护院",
        "猫捉老鼠厉害",
        "老鼠偷吃粮食",
        "蛇冬眠在洞中",
        "青蛙捉害虫",
        "壁虎爬墙很快",
        "蝎子尾巴有毒",
        "蜈蚣脚很多",
        "蜘蛛八条腿",
        "螃蟹横着走路",
        "虾在水中跳跃",
        "贝壳藏在沙里",
        "珊瑚色彩斑斓",
        "海星五个角",
        "水母透明漂浮",
        "章鱼喷墨汁",
        "乌贼变色伪装",
        "海马爸爸育儿",
        "海葵随水流动",
        "海参软软蠕动",
        "海胆浑身是刺",
        "鲍鱼味道鲜美",
        "扇贝可以食用",
        "牡蛎营养丰富",
        "蛤蜊吐沙干净",
        "螺蛳小小的",
        "蜗牛背着壳",
        "蚯蚓松土有益",
        "蚂蚱跳跃力强",
        "蟋蟀夜晚鸣叫",
        "螳螂捕食害虫",
        "蝉夏天唱歌",
        "萤火虫发光",
        "蝴蝶翅膀美丽",
        "蜜蜂勤劳采蜜",
        "蚂蚁团结协作",
        "瓢虫背上有星",
        "蜻蜓眼睛很大",
        "蚊子吸血讨厌",
        "苍蝇传播疾病",
        "蟑螂生命力强",
        "飞蛾扑向灯火",
        "蚕宝宝吐丝",
        "蝴蝶蜕变成长",
        "毛毛虫变美丽"
    ]
    
    # 分词（按字符分词，适用于中文）
    corpus = [list(sentence) for sentence in sentences]
    
    return corpus


def split_corpus(corpus: List[List[str]], train_ratio: float = 0.8) -> Tuple[List[List[str]], List[List[str]]]:
    """
    将语料库划分为训练集和测试集
    
    Args:
        corpus: 完整语料库
        train_ratio: 训练集比例
        
    Returns:
        (训练集, 测试集)
    """
    random.shuffle(corpus)
    split_idx = int(len(corpus) * train_ratio)
    train_corpus = corpus[:split_idx]
    test_corpus = corpus[split_idx:]
    return train_corpus, test_corpus


def main():
    """主函数"""
    print("=" * 60)
    print("N-Gram 语言模型演示")
    print("=" * 60)
    
    # 设置随机种子以保证可重复性
    random.seed(42)
    
    # 生成语料库
    print("\n【1. 数据生成】")
    corpus = generate_corpus()
    print(f"生成语料库共 {len(corpus)} 个句子")
    print(f"示例句子: {''.join(corpus[0])}")
    
    # 划分训练集和测试集
    train_corpus, test_corpus = split_corpus(corpus, train_ratio=0.8)
    print(f"训练集: {len(train_corpus)} 个句子")
    print(f"测试集: {len(test_corpus)} 个句子")
    
    # 测试不同的 N 值
    for n in [2, 3, 4]:
        print("\n" + "=" * 60)
        print(f"【{n}-Gram 模型】")
        print("=" * 60)
        
        # 创建并训练模型
        model = NGramModel(n=n)
        model.train(train_corpus)
        
        # 预测下一个词示例
        print(f"\n【3. 预测下一个词】")
        test_contexts = [
            ["今", "天"],
            ["天", "气"],
            ["明", "天"],
            ["早", "上"]
        ]
        
        for context in test_contexts:
            next_word, prob = model.predict_next(context)
            context_str = ''.join(context)
            print(f"  上下文 '{context_str}' -> 预测: '{next_word}' (概率: {prob:.6f})")
        
        # 生成文本
        print(f"\n【4. 文本生成】")
        print(f"生成 5 段样本文本（最大长度 15）:")
        for i in range(5):
            # 随机选择种子
            seed = random.choice(train_corpus)[:2] if random.random() > 0.5 else None
            generated = model.generate_text(seed=seed, max_length=15)
            print(f"  样本 {i+1}: {generated}")
        
        # 计算困惑度
        print(f"\n【5. 模型评估】")
        perplexity = model.calculate_perplexity(test_corpus)
        print(f"测试集困惑度 (Perplexity): {perplexity:.4f}")
        print(f"解释: 困惑度越低，模型性能越好")
        print(f"      困惑度为 {perplexity:.2f} 表示模型相当于面对一个")
        print(f"      有 {perplexity:.2f} 个等概率选择的困惑")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
