from base.logger import logger
from rag_qa.core.bm25_search import bm25
from rag_qa.core.rag_system import rag_system

class IntegratedQASystem:
    def query(self, query, source_filter=None):
        """
        主流程查询入库
        :param query:  问题
        :param source_filter: 学科
        :return: 答案
        """
        # 1) 调用redis和mysql进行查询
        answer = bm25.search(query, threshold=0.9)
        if answer:
            logger.info("MySQL或者Redis 答案")
            # True表示的是当前问题已经回答完毕
            yield answer, True
        else:
            logger.info("RAG 模型答案")
            answer = rag_system.generate_answer(query, source_filter)
            for token in answer:
                # 在for循环的过程中, 认为问题是没有回答完毕
                yield token, False
            # for循环结束,认为问题已经回答完毕
            yield "", True


iqa = IntegratedQASystem()
if __name__ == "__main__":
    # query = "Java的基础语法有哪些"
    query = "LLM大模型语言"
    answer = iqa.query(query)
    for token in answer:
        print(token)