# EduRAG

面向教育场景的 **RAG（Retrieval-Augmented Generation，检索增强生成）** 问答系统。项目支持多格式教学文档解析、中文文本切分、向量检索与大模型问答，并结合 MySQL 知识库、Redis 缓存与 Milvus 向量数据库，为学科知识问答提供完整技术栈。

## 功能特性

- **多格式文档加载**：支持 PDF、Word（`.docx`）、PPT（`.pptx`）、图片等格式，内置 OCR 识别扫描件与嵌入图片中的文字
- **中文文本切分**：提供基于标点与段落的中文递归切分器，以及基于达摩院语义模型的文档切分器
- **结构化知识库**：MySQL 存储学科问答对（`jpkb` 表）与会话历史（`conversations` 表）
- **答案缓存**：Redis 缓存高频问答，降低重复查询开销
- **向量检索**：Milvus 向量数据库支撑语义检索（配置已就绪）
- **大模型接入**：通过阿里云 DashScope 兼容接口调用通义千问（Qwen）系列模型

## 技术栈

| 类别 | 技术 |
|------|------|
| 文档处理 | LangChain、PyMuPDF、python-docx、python-pptx |
| OCR | RapidOCR（Paddle / ONNX Runtime） |
| 向量数据库 | Milvus、pymilvus、langchain-milvus |
| 大模型 | OpenAI 兼容 API（DashScope / 通义千问） |
| 关系型数据库 | MySQL（PyMySQL） |
| 缓存 | Redis |
| Web 框架 | FastAPI、Gradio、Uvicorn |
| 深度学习 | PyTorch、Transformers、Sentence-Transformers、FlagEmbedding |

## 项目结构

```
edu_rag/
├── base/                          # 基础模块
│   ├── config.py                  # 配置加载（读取 config.ini）
│   └── logger.py                  # 日志初始化
├── rag_qa/                        # RAG 核心模块
│   ├── core/                      # RAG 核心逻辑
│   │   ├── document_processor.py  # 文档加载与切分（多格式统一入口）
│   │   ├── vector_store.py        # Milvus 向量库创建、入库与混合检索
│   │   ├── query_classifier.py    # 问题分类（通用知识 / 专业知识）
│   │   ├── strategy_selector.py   # 检索策略选择（直接检索 / 假设问题 / 子查询 / 回溯）
│   │   ├── prompts.py             # 提示词模板（RAG、策略选择、HyDE 等）
│   │   └── rag_system.py          # RAG 系统主流程（分类 → 检索 → 生成）
│   ├── edu_document_loaders/      # 文档加载器（含 OCR）
│   │   ├── edu_ocr.py             # OCR 引擎封装
│   │   ├── edu_pdfloader.py       # PDF 加载器
│   │   ├── edu_docloader.py       # Word 加载器
│   │   ├── edu_pptloader.py       # PPT 加载器
│   │   └── edu_imgloader.py       # 图片加载器
│   ├── edu_text_spliter/          # 文本切分器
│   │   ├── edu_chinese_recursive_text_splitter.py  # 中文递归切分
│   │   └── edu_model_text_spliter.py               # 语义模型切分
│   ├── db_util/                   # 数据库客户端
│   │   ├── mysql_client.py        # MySQL 操作
│   │   └── redis_client.py        # Redis 操作
│   └── main.py                    # 应用入口
├── data/                          # 示例数据
│   ├── JP学科知识问答.csv         # 学科问答知识库样本
│   └── ai_data/                   # 多格式文档示例（PDF / Word / PPT / 图片）
├── test/                          # 测试与调试脚本
│   ├── 1.json
│   └── 向量模型.py
├── logs/                          # 运行日志（由 config.ini 配置路径）
├── config.ini                     # 项目配置文件
├── requirements.txt               # Python 依赖
├── superman.svg                   # 项目图标
└── README.md
```

## 环境要求

- Python 3.10+
- MySQL 5.7+ / 8.0+
- Redis 6.0+
- Milvus 2.x
- （可选）NVIDIA GPU，用于 OCR 与嵌入模型加速

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd edu_rag
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

> OCR 默认优先使用 `rapidocr_paddle`（GPU 加速）。若仅 CPU 环境，将自动回退至 `rapidocr_onnxruntime`。

### 3. 配置服务

复制并编辑 `config.ini`，填入各服务的连接信息：

```ini
[mysql]
host = localhost
user = root
password = your_password
database = subjects_kg

[redis]
host = localhost
port = 6379
password = your_password
db = 0

[milvus]
host = localhost
port = 19530
database_name = itcast
collection_name = edurag

[llm]
model = qwen-plus
dashscope_api_key = your_api_key
dashscope_base_url = https://dashscope.aliyuncs.com/compatible-mode/v1

[retrieval]
parent_chunk_size = 1200
child_chunk_size = 300
chunk_overlap = 50
retrieval_k = 3
candidate_m = 2

[app]
valid_sources = ["ai", "java", "test", "ops", "bigdata"]
customer_service_phone = 13000000000

[logger]
log_file = logs/app.log
```

### 4. 初始化数据库

在 MySQL 中创建数据库并建表：

```sql
CREATE DATABASE IF NOT EXISTS subjects_kg DEFAULT CHARSET utf8mb4;

USE subjects_kg;

-- 学科问答知识库
CREATE TABLE IF NOT EXISTS jpkb (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(100) NOT NULL COMMENT '学科名称',
    question TEXT NOT NULL COMMENT '问题',
    answer TEXT NOT NULL COMMENT '答案'
);

-- 会话历史
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL COMMENT '会话 ID',
    question TEXT NOT NULL COMMENT '用户问题',
    answer TEXT NOT NULL COMMENT '系统回答',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '时间戳',
    INDEX idx_session_id (session_id)
);
```

### 5. 导入示例数据

```bash
python -m rag_qa.db_util.mysql_client
```

该脚本会将 `data/JP学科知识问答.csv` 中的问答数据写入 `jpkb` 表。

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `parent_chunk_size` | 父文档块大小（字符数） | 1200 |
| `child_chunk_size` | 子文档块大小（字符数） | 300 |
| `chunk_overlap` | 块重叠大小 | 50 |
| `retrieval_k` | 向量检索返回数量 | 3 |
| `candidate_m` | 重排序后保留的候选数量 | 2 |
| `valid_sources` | 允许的知识来源学科列表 | ai, java, test, ops, bigdata |

## 模块使用示例

### 文档加载
<img width="2720" height="760" alt="image" src="https://github.com/user-attachments/assets/1669c382-7cc1-4a1a-bf69-63ba85d671c6" />


```python
from rag_qa.edu_document_loaders.edu_pdfloader import OCRPDFLoader
from rag_qa.edu_document_loaders.edu_docloader import OCRDOCLoader
from rag_qa.edu_document_loaders.edu_pptloader import OCRPPTLoader
from rag_qa.edu_document_loaders.edu_imgloader import OCRIMGLoader

# PDF
pdf_docs = OCRPDFLoader("path/to/document.pdf").load()

# Word
docx_docs = OCRDOCLoader("path/to/document.docx").load()

# PPT
ppt_docs = OCRPPTLoader("path/to/slides.pptx").load()

# 图片
img_docs = OCRIMGLoader("path/to/image.png").load()
```

### 中文文本切分
<img width="2436" height="1366" alt="image" src="https://github.com/user-attachments/assets/cf2b6020-f31c-47c1-ae3c-e6ae0c831a3c" />
<img width="2519" height="417" alt="image" src="https://github.com/user-attachments/assets/d0ebfaba-224f-4e11-924f-26d1ee7f7194" />


```python
from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter

splitter = ChineseRecursiveTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    keep_separator=True,
)
chunks = splitter.split_text("你的长文本内容...")
```

### MySQL 知识库查询

```python
from rag_qa.db_util.mysql_client import MySQLClient

client = MySQLClient()
answer = client.fetch_answer("用上下文管理器实现函数运行时间的计算?")
history = client.fetch_recent_history("session-001")
client.close()
```

### Redis 缓存

```python
from rag_qa.db_util.redis_client import RedisClient

redis = RedisClient()
redis.set_data("answer:某个问题", "缓存的答案")
cached = redis.get_answer("某个问题")
```

## 数据格式

`data/JP学科知识问答.csv` 为知识库导入样本，字段如下：

| 字段 | 说明 |
|------|------|
| 学科名称 | 所属学科，如 Python学科、Java学科 |
| 问题 | 问答对中的问题 |
| 答案 | 对应的标准答案 |

## 系统架构

```
用户提问
    │
    ▼
┌─────────────┐     命中      ┌─────────────┐
│ Redis 缓存  │ ────────────► │  直接返回答案 │
└─────────────┘               └─────────────┘
    │ 未命中
    ▼
┌─────────────┐     命中      ┌─────────────┐
│ MySQL 精确匹配│ ────────────► │  返回答案     │
└─────────────┘               └─────────────┘
    │ 未命中
    ▼
┌─────────────┐
│ Milvus 向量检索│
└─────────────┘
    │
    ▼
┌─────────────┐
│ LLM 生成回答  │ ──► 写入 Redis / MySQL 会话历史
└─────────────┘
```

## 注意事项

1. **API 密钥安全**：请勿将真实的 `dashscope_api_key` 提交到版本库，建议使用环境变量或本地配置文件管理
2. **OCR 性能**：PDF 中仅对超过页面宽高 60% 阈值的图片执行 OCR，以提升非扫描版 PDF 的处理速度
3. **语义切分模型**：`AliTextSplitter` 依赖 ModelScope 文档分割模型，首次使用需下载模型权重，默认在 CPU 上运行
4. **日志路径**：`config.ini` 中 `log_file` 支持相对路径（相对于项目根目录）或绝对路径

## 许可证

本项目仅供学习与研究使用。
