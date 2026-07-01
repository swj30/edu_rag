# 导入配置解析库
import configparser
# 导入路径操作库
from pathlib import Path


class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file=None):
        # 项目根目录
        self.base_dir = Path(__file__).resolve().parent.parent
        # 解析配置文件路径
        config_path = Path(config_file) if config_file else self.base_dir / 'config.ini'
        if not config_path.is_absolute():
            config_path = self.base_dir / config_path
        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_path, encoding='utf-8')

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        # MySQL 用户名
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # MySQL 密码
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='1234')
        # MySQL 数据库名
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # Redis 端口
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='123456')
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        # Milvus 端口
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag')

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key',fallback="sk-71f17dd2e54c45c8af4a9624af85cd79")
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')
        # 日志文件路径
        log_file = self.config.get('logger', 'log_file', fallback='logs/app.log')
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = self.base_dir / log_path
        self.LOG_FILE = str(log_path)


# 创建一个Config对象
config = Config()

if __name__ == '__main__':
    conf = Config()
    print(conf.CHILD_CHUNK_SIZE)
