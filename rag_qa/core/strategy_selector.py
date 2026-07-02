# 本模块负责根据问题选择合适检索策略
from openai import OpenAI

from rag_qa.core.prompts import RAGPrompts
from base.config import config as conf
from base.logger import logger
from rag_qa.core.vector_store import vector_store


class StrategySelector:
    def __init__(self):
        # 创建 OpenAI 客户端
        self.client = OpenAI(
            api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL
        )

    # 根据用户问题 选择检索策略 进行检索
    def strategy_select(self, query, source_filter):
        # 生成大模型分类提示词
        prompt = RAGPrompts.strategy_select_prompt().format(query=query)
        # 调用大模型生成策略
        strategy = self.call_dashscope(prompt)
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")

        # 根据检索策略选择不同的检索方式
        ranked_sub_chunks = []  # 初始化
        if strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = vector_store.hybrid_search_with_reranker(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )

        final_context_docs = ranked_sub_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs


    # 定义私有方法，使用假设文档进行检索
    def _retrieve_with_hyde(self, query):
        logger.info(f"使用 假设文档 策略进行检索 (查询: '{query}')")
        # 获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分
        try:
            # 调用大语言模型生成假设答案
            hypo_answer = self.call_dashscope(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"生成的假设答案: '{hypo_answer}'")
            # 使用假设答案进行检索，并返回检索结果
            return vector_store.hybrid_search_with_reranker(
                hypo_answer, k=conf.RETRIEVAL_K
            )
        except Exception as e:
            logger.error(f"假设性 策略执行失败: {e}")
            return []

    # 定义私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query):
        logger.info(f"使用 子查询  策略进行检索 (查询: '{query}')")
        # 获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            # 调用大语言模型生成子查询列表
            subqueries_text = self.call_dashscope(
                subquery_prompt_template.format(query=query)
            ).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            all_docs = []  # 初始化空列表，用于存储所有子查询的检索结果
            #   遍历每个子查询
            for sub_q in subqueries:
                # 使用子查询进行检索，并将结果添加到列表中
                docs = vector_store.hybrid_search_with_reranker(
                    sub_q, k=conf.RETRIEVAL_K
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            # 对所有检索结果进行去重
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())

            logger.info(
                f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个"
            )
            return unique_docs
        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    # 调用阿里的API接口
    def call_dashscope(self, prompt):
        # 调用 DashScope API
        try:
            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            # 返回完成结果
            return (
                completion.choices[0].message.content
                if completion.choices
                else "直接检索"
            )
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"


ss = StrategySelector()

if __name__ == '__main__':
    # print(ss.strategy_select("JAVA的课程大纲是什么？","ai"))
    print(ss.strategy_select("JAVA的课程大纲和Python大纲的区别是什么？","ai"))