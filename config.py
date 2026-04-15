#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

统一管理模型参数、路径、随机种子等配置项
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class SmoothingMethod(Enum):
    """平滑方法枚举"""
    ADD_ONE = "add_one"           # Add-1 (Laplace) 平滑
    ADD_K = "add_k"               # Add-k 平滑
    KNESER_NEY = "kneser_ney"     # Kneser-Ney 平滑


class SamplingStrategy(Enum):
    """采样策略枚举"""
    WEIGHTED_RANDOM = "weighted_random"   # 加权随机采样
    GREEDY = "greedy"                     # 贪婪采样（选概率最大）
    BEAM_SEARCH = "beam_search"           # 束搜索


@dataclass
class ModelConfig:
    """
    模型配置类

    Attributes:
        n: N-Gram 的阶数，默认为 2 (bigram)
        smoothing: 平滑方法，默认为 Add-1
        smoothing_k: Add-k 平滑的参数 k，默认为 1.0
    """
    n: int = 2
    smoothing: SmoothingMethod = SmoothingMethod.ADD_ONE
    smoothing_k: float = 1.0

    def __post_init__(self):
        """初始化后验证参数"""
        if self.n < 1:
            raise ValueError(f"N-Gram 阶数 n 必须 >= 1，当前值: {self.n}")
        if self.smoothing_k <= 0:
            raise ValueError(f"平滑参数 k 必须 > 0，当前值: {self.smoothing_k}")


@dataclass
class GenerationConfig:
    """
    文本生成配置类

    Attributes:
        max_length: 生成文本的最大长度
        sampling_strategy: 采样策略
        beam_width: 束搜索的宽度（仅在 beam_search 策略下使用）
        temperature: 温度参数，控制生成的随机性（越高越随机）
    """
    max_length: int = 20
    sampling_strategy: SamplingStrategy = SamplingStrategy.WEIGHTED_RANDOM
    beam_width: int = 3
    temperature: float = 1.0

    def __post_init__(self):
        """初始化后验证参数"""
        if self.max_length < 1:
            raise ValueError(f"最大长度必须 >= 1，当前值: {self.max_length}")
        if self.beam_width < 1:
            raise ValueError(f"束搜索宽度必须 >= 1，当前值: {self.beam_width}")
        if self.temperature <= 0:
            raise ValueError(f"温度参数必须 > 0，当前值: {self.temperature}")


@dataclass
class DataConfig:
    """
    数据配置类

    Attributes:
        train_ratio: 训练集比例
        random_seed: 随机种子
        start_token: 句子起始标记
        end_token: 句子结束标记
    """
    train_ratio: float = 0.8
    random_seed: int = 42
    start_token: str = "<START>"
    end_token: str = "<END>"

    def __post_init__(self):
        """初始化后验证参数"""
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"训练集比例必须在 (0, 1) 之间，当前值: {self.train_ratio}")


@dataclass
class LoggingConfig:
    """
    日志配置类

    Attributes:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        format: 日志格式
        file_path: 日志文件路径（None 表示只输出到控制台）
        console_output: 是否输出到控制台
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    console_output: bool = True

    def __post_init__(self):
        """初始化后验证参数"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是 {valid_levels} 之一，当前值: {self.level}")


@dataclass
class CorpusConfig:
    """
    语料库配置类

    Attributes:
        sentences: 预定义的语料句子列表
    """
    sentences: List[str] = field(default_factory=lambda: [
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
    ])


# 默认配置实例
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_GENERATION_CONFIG = GenerationConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_LOGGING_CONFIG = LoggingConfig()
DEFAULT_CORPUS_CONFIG = CorpusConfig()
