# 本模块用于对用户提出的问题进行分类,分为通用知识和专业知识

import os
import torch  # 导入PyTorch深度学习框架
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)  # 导入BERT分词器和序列分类模型
from base.logger import logger  # 导入日志记录器


class QueryClassifier:
    def __init__(self, model_dir):
        if not os.path.exists(model_dir):  # 检查模型目录是否存在
            raise FileNotFoundError(
                f"模型目录不存在: {model_dir}"
            )  # 如果不存在则抛出异常
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # 自动选择GPU或CPU设备
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)  # 加载BERT分词器
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir
        )  # 加载BERT分类模型
        self.model.to(self.device)  # 将模型移动到指定设备（GPU或CPU）
        self.model.eval()  # 设置模型为评估模式（禁用dropout等训练特性）
        logger.info(f"模型加载成功，使用设备: {self.device}")  # 记录模型加载成功的日志

    def predict_category(self, query):
        """
        根据问题来判断是专业咨询还是通用知识
        :param query:  问题
        :return: 专业咨询  通用知识
        """
        try:
            encoding = self.tokenizer(
                query,
                truncation=True,
                padding=True,  # 对查询文本进行分词和编码，启用截断和填充
                max_length=128,
                return_tensors="pt",
            )  # 最大长度128，返回PyTorch张量
            encoding = {
                k: v.to(self.device) for k, v in encoding.items()
            }  # 将编码结果移动到指定设备
            with torch.no_grad():  # 禁用梯度计算以节省内存和加速推理
                outputs = self.model(**encoding)  # 使用模型进行前向传播预测
                prediction = torch.argmax(
                    outputs.logits, dim=1
                ).item()  # 获取logits最大值对应的类别索引
            return (
                "专业咨询" if prediction == 1 else "通用知识"
            )  # 根据预测结果返回类别名称（1=专业咨询，0=通用知识）
        except Exception as e:  # 捕获所有异常
            logger.error(f"预测失败: {e}")  # 记录错误日志
            return "通用知识"  # 发生错误时返回默认类别


classifier = QueryClassifier(r"D:/workspace/pyproject/models/bert_results")

if __name__ == "__main__":
    querys = ["springboot的原理是什么", "AI学科的学费是多少", "大语言模型"]  # 测试查询
    for query in querys:
        category = classifier.predict_category(query)  # 进行预测
        print(f"查询类别: {category}")