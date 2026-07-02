# 本模块负责向量库的创建和文本向量化入库和问题的向量搜索
import hashlib

from langchain_core.documents import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from sentence_transformers import CrossEncoder

from base.config import Config
from base.logger import logger
from rag_qa.core.document_processor import load_documents_from_directory, process_documents

conf = Config()


# 向量化存储
class VectorStore:

    def __init__(self):
        # 1) 初始化向量模型
        # 自动检测设备：优先使用 CUDA，否则使用 CPU
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        # 指定模型位置
        bge_m3_model_path = "D:/workspace/pyproject/models/bge-m3"

        # 创建本地模型 bge-m3
        self.embedding = BGEM3EmbeddingFunction(model_name_or_path=bge_m3_model_path, use_fp16=True, device=device)

        # 创建重排序模型
        self.reranker = CrossEncoder(model_name_or_path="D:/workspace/pyproject/models/bge-reranker-large")

        # 2) 初始化Milvus客户端
        self.client = MilvusClient(
            uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}",
            db_name=conf.MILVUS_DATABASE_NAME
        )

        # 3) 创建Milvus数据表
        if not self.client.has_collection(conf.MILVUS_COLLECTION_NAME):
            # 创建集合
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加字段 id，作为主键 VARCHAR类型，最大长度100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding.dim['dense'])
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",  # IVF_FLAT索引是一种可以提高浮点向量搜索性能的索引算法
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(
                collection_name=conf.MILVUS_COLLECTION_NAME,
                schema=schema,
                index_params=index_params
            )
            # 记录创建集合的日志
            logger.info(f"已创建集合 {conf.MILVUS_COLLECTION_NAME}")
        else:
            logger.info(f"集合 {conf.MILVUS_COLLECTION_NAME} 已存在")

    def add_documents(self, documents: [Document]):
        """
        将文档向量化之后存储到Milvus
        :param documents: 要存储的文档集合
        :return: 无
        """
        texts = [document.page_content for document in documents]  # []
        embeddings = self.embedding(texts)

        data = []  # 存储最后想向量库保存的内容

        # 遍历集合，拿到每个文档
        for i, document in enumerate(documents):

            # 生成子块对应的稀疏向量
            # 初始化稀疏向量字典
            sparce_vector = {}
            row = embeddings['sparse'][[i]]  # 稀疏向量
            indices = row.indices  # 获取稀疏向量的非零索引值
            values = row.data  # 获取稀疏向量的非零值
            # 将索引和值配对，填充到稀疏向量字典中
            for index, value in zip(indices, values):  # (0,100)  0.12222222    {"100":0.12222}
                sparce_vector[index] = value

            data.append({
                "id": hashlib.md5(document.page_content.encode("utf-8")).hexdigest(),  # 主键： 子块内容的md5值
                "text": document.page_content,  # 子块内容
                "dense_vector": embeddings['dense'][i],  # 稠密向量
                "sparse_vector": sparce_vector,  # 稀疏向量
                "parent_id": document.metadata["parent_id"],  # 父块ID:
                "parent_content": document.metadata["parent_content"],  # 父块内容
                "source": document.metadata["source"],  # 来源
                "timestamp": document.metadata["timestamp"],  # 时间戳
            })

        # 将文档的集合插入到Milvus
        if data:
            self.client.upsert(
                collection_name=conf.MILVUS_COLLECTION_NAME,
                data=data
            )

    def hybrid_search_with_reranker(self, query, k=conf.RETRIEVAL_K, source_filter=None) -> list[Document]:
        """
            根据问题从知识库中查询符合条件的父块内容
            :param query: 查询问题
            :param k: 稠密和稀疏检索要返回的子块数量
            :param source_filter: 查询是哪个学科的内容， 如图不传， 默认所有学科
            :return: 匹配中的父块内容
        """

        # 1. 查询条件向量化(提取稠密向量, 稀疏向量)
        embddings = self.embedding([query])
        dense_query_vector = embddings['dense'][0]  # 获取查询问题的稠密向量
        sparse_query_vector = {}  # 获取查询问题的稀疏向量
        row = embddings["sparse"][[0]]
        index = row.indices
        value = row.data
        for idx, val in zip(index, value):
            sparse_query_vector[idx] = val

        #print(f"稠密向量: {dense_query_vector}")  # [-0.02619578 -0.03766342 -0.02455433 ...  0.00410029  0.01828877 0.01672264]
        #print(f"稀疏向量: {sparse_query_vector}")  # {6: 0.05004845932126045, 573: 0.14645037055015564, ...}

        # 2. 构建混合搜索请求参数

        # 初始化过滤表达式   "source == 'ai'"
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # 创建多个AnnSearchRequest实例
        # 稠密向量
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 稀疏向量
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 配置重排序策略
        ranker = WeightedRanker(0.7, 1.0)

        # 3. 进行混合搜索
        # return [id, distance, entity{}]
        results = self.client.hybrid_search(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            reqs=[sparse_request, dense_request],
            ranker=ranker,
            limit=k,
            output_fields = ["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]

        # print(results)

        # 4. 把搜索的结果封住成Document对象
        chunks = [self.__doc_from_hit(result["entity"]) for result in results]  # chunks -> list[Document]
        # print(chunks)

        # 5. 把检索出的父文档内容进行去重
        parent_docs = self.__get_unique_parent_docs(chunks) # parent_docs -> [Document]
        # print(parent_docs)

        # 6. 判断父文档的数量
        if len(parent_docs) < 2:
            # 6.1. 如果数量<2直接返回
            return parent_docs
        else:
            # 6.2. 如果父文档的数量>= 2 调用排序模型，进行重排序后返回
            # 创建查询与文档的内容配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用重排序器模型，重新排序
            scores = self.reranker.predict(pairs)  # [0.9963676 0.8428106]
            # 根据得分从高到低排序文档
            ranked_parent_docs = [
                doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)
            ]
            # 返回前 m 个重排序后的文档
            print(ranked_parent_docs)
            return ranked_parent_docs[: conf.CANDIDATE_M]


    # 定义私有方法，从 Milvus 查询结果创建 Document 对象
    def __doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )

    # 定义私有方法，从子块中提取去重的父文档
    def __get_unique_parent_docs(self, chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content")
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs


# 创建向量存储实例
vector_store = VectorStore()

if __name__ == '__main__':
    # 加载 拆分文档
    # documents = load_documents_from_directory("D:/workspace/pyproject/edu_rag/data/ai_data")
    # chunks = process_documents(documents)
    # 向量化并保存
    # vector_store.add_documents(chunks)
    query = "什么是大语言模型"
    vector_store.hybrid_search_with_reranker(query)
