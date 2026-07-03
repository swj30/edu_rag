# 本模块负责RAG系统的调用主流程
from openai import OpenAI

from rag_qa.core.query_classifier import classifier
from rag_qa.core.strategy_selector import ss
from rag_qa.core.vector_store import vector_store
from base.config import config as conf
from rag_qa.core.prompts import RAGPrompts
from base.logger import logger






class RAGSystem:

    def __init__(self):
        # 初始化大模型
        self.client = OpenAI(
            api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL
        )



    def generate_answer(self, query, source_filter=None, history=None) -> str:
        """
            根据用户的问题调用大模型生成答案
            :param query: 问题
            :param source_filter: 学科
            :return: 问题的答案
        """

        # 获取历史记录, 拼接为字符串
        if history is None:
            history = []
        else:
            history = history[:5]
        # 历史记录拼接为字符串
        history_context = "\n".join(
            [f"Q:{i['question']}\nA:{i['answer']}" for i in history]
        )

        # 1. 调用Bert分类模型得到用户问题类型 通用知识/专业咨询
        # category = classifier.predict_category(query)
        # 写死吧，训练的大模型不太靠谱
        category =  "专业咨询"
        print(f"查询类别: {category}")

        # 2. 根据分类结果生成提示词中的上下文内容
        if category == "通用知识":
            # 2.1 如果查询类别是通用知识，content = ""
            context = ""
        elif category == "专业咨询":
            # 2.2 如果查询类别是专业查询，content为向量数据库搜索出的东西
            # document = vector_store.hybrid_search_with_reranker(query=query, source_filter=source_filter)
            # context = "\n\n".join([doc.page_content for doc in document])

            """
                多策略提示词
            """
            # 先调用大模型对查询的问题进行分类： 1-直接检索 2-假设问题检索 3-子查询检索
            documents = ss.strategy_select(query, source_filter)
            context = "\n\n".join([doc.metadata["parent_content"] for doc in documents])
        else:
            raise Exception("无法识别的类型")



        # 3. 组装提示词发送给大模型生成答案
        try:
            # 组装提示词
            print(f"上下文: {context}")
            prompt = RAGPrompts.rag_prompt().format(
                context=context,
                history=history,
                question=query,
                phone=conf.CUSTOMER_SERVICE_PHONE
            )
            logger.info(f"发送LLM的提示词是:{prompt}")
            # 调用大模型生成答案
            completion = self.client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                timeout=30,  # 设置 30 秒超时
                stream=True,  # 启用流式输出
            )
            # 遍历流式输出的每个 chunk
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 获取当前 chunk 的内容
                    content = chunk.choices[0].delta.content
                    # 逐 token 返回，供前端实时显示
                    yield content
        except Exception as e:
            logger.error(e)
            answer = f"抱歉，处理您的知识问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

rag_system = RAGSystem()

if __name__ == '__main__':
    answer = rag_system.generate_answer("LLM背景知识", source_filter="ai", session_id="101")
    print(answer)