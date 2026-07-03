from base.logger import logger
from rag_qa.core.bm25_search import bm25
from rag_qa.core.rag_system import rag_system
from rag_qa.db_util.mysql_client import mysql_client


class IntegratedQASystem:
    def query(self, query, source_filter=None, session_id=None):
        """
        主流程查询入库
        :param query:  问题
        :param source_filter: 学科
        :return: 答案
        """
        if session_id is None:
            raise Exception("会话id不能为空")


        # 1) 调用redis和mysql进行查询
        answer = bm25.search(query, threshold=0.9)
        if answer:
            logger.info("MySQL或者Redis 答案")
            # 更新会话历史记录
            mysql_client.update_session_history(session_id, query, answer)
            # True表示的是当前问题已经回答完毕
            yield answer, True
        else:
            logger.info("RAG 模型答案")

            # 根据会话id查询会话历史记录
            history = mysql_client.fetch_recent_history(session_id)

            collected_answer = ""
            answer = rag_system.generate_answer(query, source_filter=source_filter, history = history)
            for token in answer:

                collected_answer += token
                # 在for循环的过程中, 认为问题是没有回答完毕
                yield token, False

            # 更新历史会话记录
            mysql_client.update_session_history(session_id,query, collected_answer)
            # for循环结束,认为问题已经回答完毕
            yield "", True


iqa = IntegratedQASystem()
if __name__ == "__main__":
    # query = "Java的基础语法有哪些"
    query = "Java的基础语法有哪些"
    answer = iqa.query(query, session_id="101")
    for token in answer:
        print(token)