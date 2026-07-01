# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示

from milvus_model.hybrid import BGEM3EmbeddingFunction

from base.logger import logger


def create_vector_store():
    # 自动检测设备：优先使用 CUDA，否则使用 CPU
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"使用设备: {device}")

    # 指定模型位置
    bge_m3_model_path = "E:/1期/资料/edurag/models/bge-m3"

    # 创建本地模型 bge-m3
    embedding_function = BGEM3EmbeddingFunction(model_name_or_path=bge_m3_model_path, use_fp16=False, device=device)
    print(f"嵌入向量的维度：{embedding_function.dim['dense']}")

    # '''
    #     BGE-M3 是多向量嵌入模型，常见有两种：
    #     dense（密集向量）：用于语义相似度检索
    #     sparse（稀疏向量）：用于词汇匹配检索
    # '''
    #
    # 测试文本
    # texts = ['今天北京天气挺好的', '北京天气是晴天', '上海水上交通堵了']
    texts = ['今天北京天气挺好的']

    # 使用bge-m3嵌入生成文档向量
    embeddings = embedding_function(texts)

    for i, text in enumerate(texts):
        logger.info(f"文档 {i} 的稠密向量：{embeddings['dense'][i]}")
        logger.info(f"文档 {i} 的稀疏向量：{embeddings['sparse'][[i]]}")
        print("============================")


if __name__ == "__main__":
    create_vector_store()