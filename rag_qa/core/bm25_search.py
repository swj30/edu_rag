import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from base.logger import logger
from rag_qa.db_util.mysql_client import mysql_client
from rag_qa.db_util.redis_client import redis_client



class BM25Search:

    def __init__(self):
        """
            初始化
        """
        # 从数据库中查询所有问题
        self.original_questions = mysql_client.fetch_questions()
        # 对数据进行分词
        tokenized_questions = [jieba.lcut(q[0]) for q in self.original_questions]
        # 初始化BM25模型
        self.bm25 = BM25Okapi(tokenized_questions)
        logger.info("BM25模型初始化成功")


    def search(self,query,threshold=0.9):
        """
        查询问题
        :param query: 问题
        :param threshold: 相似度匹配阈值
        :return: 搜索的结果
        """

        # 1. 先查Redis缓存
        cache_answer = redis_client.get_answer(f"answer:{query}")
        if cache_answer:
            # 命中缓存
            logger.info(f"从缓存中获取{query}的答案")
            return cache_answer

        # 2. Redis缓存中没有，进行BM25从数据库中搜索
        # 2,1 获取分数
        scores = self.bm25.get_scores(jieba.lcut(query))
        # 2.2 归一化处理
        softmax_scores = self._softmax(scores)
        # 2.3 获取分数最大的问题索引
        best_index = softmax_scores.argmax()
        # 2.4 获取最大的分数
        best_score = softmax_scores[best_index]

        # 3. 获的最高分数跟阈值比较
        if best_score > threshold:
            # 3.1 如果大于阈值，拿到问题，从数据库中查询出答案
            question = self.original_questions[best_index]
            answer = mysql_client.fetch_answer(question)
            # 如果查询出来了答案，加入到缓存中
            if answer:
                redis_client.set_data(f"answer:{query}", answer)
            return answer
        else:
            # 3.2 如果不大于阈值，返回None,方便后续判断走RAG
            logger.info(f"没有找到答案")
            return None


    # 归一化函数
    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()





bm25 = BM25Search()
if __name__ == '__main__':
    answer = bm25.search("两个人开发项目")
    print(answer)