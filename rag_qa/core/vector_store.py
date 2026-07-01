# 本模块负责向量库的创建和文本向量化入库和问题的向量搜索
import hashlib

from langchain_core.documents import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType
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
                index_type="IVF_FLAT",
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
        texts = [document.page_content for document in documents] #[]
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


# 创建向量存储实例
vector_store = VectorStore()

if __name__ == '__main__':
    # 加载 拆分文档
    documents = load_documents_from_directory("D:/workspace/pyproject/edu_rag/data/ai_data")
    chunks = process_documents(documents)
    # 向量化并保存
    vector_store.add_documents(chunks)