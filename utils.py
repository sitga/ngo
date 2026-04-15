#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块 - 语料生成、数据集划分、日志配置、异常类等
"""
import logging
import random
from typing import List, Tuple, Optional

from config import LogConfig, DEFAULT_LOG_CONFIG


class NGramError(Exception):
    """N-Gram 模型基异常类"""
    pass


class EmptyCorpusError(NGramError):
    """空语料库异常"""
    pass


class InvalidContextError(NGramError):
    """无效上下文异常"""
    pass


class InvalidInputError(NGramError):
    """无效输入异常"""
    pass


class VocabularyError(NGramError):
    """词汇表异常"""
    pass


def setup_logger(
    name: str,
    log_config: Optional[LogConfig] = None
) -> logging.Logger:
    """
    配置并获取日志记录器
    
    Args:
        name: 日志记录器名称
        log_config: 日志配置对象
        
    Returns:
        配置好的日志记录器
    """
    log_config = log_config or DEFAULT_LOG_CONFIG
    logger = logging.getLogger(name)
    
    level = getattr(logging, log_config.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(log_config.log_format)
    
    if log_config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    if log_config.log_file:
        file_handler = logging.FileHandler(log_config.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    return logger


def generate_corpus() -> List[List[str]]:
    """
    生成模拟语料库（关于天气和日常对话）
    
    Returns:
        分词后的语料库列表
    """
    sentences = [
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
    
    corpus = [list(sentence) for sentence in sentences]
    return corpus


def split_corpus(
    corpus: List[List[str]],
    train_ratio: float = 0.8,
    random_seed: Optional[int] = None
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    将语料库划分为训练集和测试集
    
    Args:
        corpus: 完整语料库
        train_ratio: 训练集比例 (0-1)
        random_seed: 随机种子，保证可重复性
        
    Returns:
        (训练集, 测试集)
        
    Raises:
        InvalidInputError: 语料库为空或比例无效
    """
    if not corpus:
        raise InvalidInputError("语料库不能为空")
    
    if not (0 < train_ratio < 1):
        raise InvalidInputError(f"训练集比例必须在 (0, 1) 范围内，当前值: {train_ratio}")
    
    if random_seed is not None:
        random.seed(random_seed)
    
    shuffled_corpus = corpus.copy()
    random.shuffle(shuffled_corpus)
    
    split_idx = int(len(shuffled_corpus) * train_ratio)
    train_corpus = shuffled_corpus[:split_idx]
    test_corpus = shuffled_corpus[split_idx:]
    
    return train_corpus, test_corpus


def validate_corpus(corpus: List[List[str]]) -> None:
    """
    验证语料库输入有效性
    
    Args:
        corpus: 待验证的语料库
        
    Raises:
        EmptyCorpusError: 语料库为空
        InvalidInputError: 输入格式不正确
    """
    if not corpus:
        raise EmptyCorpusError("语料库不能为空")
    
    if not isinstance(corpus, list):
        raise InvalidInputError(f"语料库必须是列表类型，当前类型: {type(corpus)}")
    
    for i, sentence in enumerate(corpus):
        if not isinstance(sentence, list):
            raise InvalidInputError(f"句子 {i} 必须是列表类型，当前类型: {type(sentence)}")
        
        for j, token in enumerate(sentence):
            if not isinstance(token, str):
                raise InvalidInputError(
                    f"句子 {i} 的第 {j} 个词必须是字符串类型，当前类型: {type(token)}"
                )
