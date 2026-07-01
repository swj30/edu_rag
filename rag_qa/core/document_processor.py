# 本模块负责文档的加载和拆分
import os
from datetime import datetime

from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from rag_qa.edu_document_loaders.edu_docloader import OCRDOCLoader
from rag_qa.edu_document_loaders.edu_imgloader import OCRIMGLoader
from rag_qa.edu_document_loaders.edu_pdfloader import OCRPDFLoader
from rag_qa.edu_document_loaders.edu_pptloader import OCRPPTLoader

from base.logger import logger
from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from base.config import config as conf

# 定义不同文档加载需要使用的加载器
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}


def load_documents_from_directory(directory_path: str) -> [Document]:
    """
    加载指定目录中的文档，放入一个集合中
    :param directory_path:  文档目录
    :return: 文档的集合
    Document的数据格式：
        page_content="Hello, world!",    metadata={"source": "https://example.com"}
    """

    # for root , _ ,files in os.walk(directory_path): 递归遍历一个目录(包含子目录)下的所有文件
    # root: 当前正在遍历的目录
    # _: 当前目录下的子目录
    # files: 当前目录下的子文件

    # 获取一下当前目录名称
    source = os.path.basename(directory_path).split("_")[0]

    # 0. 声明一个文档的集合 用来收集所有加载到的文档
    documents = []

    # 1. 递归遍历指定目录下的所有文件
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            # 2. 获取文件的绝对路径
            file_path = os.path.join(root, file_name)

            # 3. 判断当前文件的类型是否是系统支持
            file_type = os.path.splitext(file_name)[1]  # 。ppt
            if file_type not in list(
                    document_loaders.keys()):  # ['.txt', '.pdf', '.docx', '.ppt', '.pptx', '.jpg', '.png', '.md']
                logger.warning(f"不支持的文件类型：{file_type}，文件：{file_path}")
                continue

            # 4. 创建文档加载器
            loader_class = document_loaders[file_type]
            if file_type == ".txt":
                loader = loader_class(file_path, encoding="utf-8")
            else:
                loader = loader_class(file_path)
            docs = loader.load()

            # 5 遍历文档， 为每个文档添加元数据
            for doc in docs:
                doc.metadata["source"] = source
                doc.metadata["file_path"] = file_path
                doc.metadata["file_type"] = file_type
                doc.metadata["timestamp"] = datetime.now().isoformat()
            # 6 将得到的集合中的文档放入到收集用的集合
            documents.extend(docs)

    # 返回
    return documents



def process_documents(documents: [Document]) -> [Document]:
    """
    对文档进行拆分
    :param documents: 未拆分文档的集合
    :return: 拆分后的文档集合
    """

    # 准备文档拆分器
    # 初始化普通文件分割器
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=conf.PARENT_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=conf.CHILD_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP)

    # 初始化markdown文件分割器
    md_parent_splitter = MarkdownTextSplitter(chunk_size=conf.PARENT_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP)
    md_child_splitter = MarkdownTextSplitter(chunk_size=conf.CHILD_CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP)

    # 准备一个集合 来收集所有的子块文档
    chunks = []

    # 遍历所有的文档
    for i, doc in enumerate(documents):
        # 获取到了一个doc文档（metadata ，page_content）
        is_md = doc.metadata["file_type"] == ".md"
        parent_splitter_to_use = md_parent_splitter if is_md else parent_splitter
        child_splitter_to_use = md_child_splitter if is_md else child_splitter
        logger.info(f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_md else 'ChineseRecursive'}")

        # 开始切割父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        for j, parent_doc in enumerate(parent_docs):
            # 为父块声明一个id
            parent_doc_id = f"doc_{i}_parent_{j}"

            # 针对每个父块，开始切割子块
            child_docs = child_splitter_to_use.split_documents([parent_doc])
            for k, child_doc in enumerate(child_docs):
                # 为子块声明一个id
                child_doc_id = f"{parent_doc_id}_child_{k}"

                # 为子块添加元数据
                child_doc.metadata["id"] = child_doc_id
                child_doc.metadata["parent_id"] = parent_doc_id
                child_doc.metadata["parent_content"] = parent_doc.page_content
                chunks.append(child_doc)

    # 返回子块集合
    return chunks


if __name__ == '__main__':
    documents = load_documents_from_directory("D:/workspace/pyproject/edu_rag/data/ai_data")
    chunks = process_documents( documents)
    for chunk in chunks:
        print(chunk)
        print("=======================")