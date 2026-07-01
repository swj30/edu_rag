# 本目录负责redis的操作
import json

import redis

from base.config import Config
from base.logger import logger

conf = Config()


class RedisClient:
    # 连接 Redis
    def __init__(self):
        try:
            self.client = redis.StrictRedis(
                host=conf.REDIS_HOST,
                port=conf.REDIS_PORT,
                password=conf.REDIS_PASSWORD,
                db=conf.REDIS_DB,
                decode_responses=True,
            )
            # 记录连接成功
            logger.info("Redis 连接成功")
        except redis.RedisError as e:
            # 记录连接失败
            logger.error(f"Redis 连接失败: {e}")
            raise

    # 存储数据到 Redis
    def set_data(self, key, value):
        try:
            # 存储 JSON 数据
            self.client.set(key, value)
            # 记录存储成功
            logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            # 记录存储失败
            logger.error(f"Redis 存储失败: {e}")

    # 从 Redis 获取数据
    def get_data(self, key):
        try:
            # 获取数据
            data = self.client.get(key)
            # 返回解析后的 JSON 数据或 None
            return data if data else None
        except redis.RedisError as e:
            # 记录获取失败
            logger.error(f"Redis 获取失败: {e}")
            # 返回 None
            return None

    # 获取查询的缓存答案
    def get_answer(self, query):
        try:
            # 从 Redis 获取答案
            answer = self.client.get(f"answer:{query}")
            if answer:
                # 记录获取成功
                logger.info(f"从 Redis 获取答案: {query}")
                # 返回答案
                return answer
            # 返回 None
            return None
        except redis.RedisError as e:
            # 记录查询失败
            logger.error(f"Redis 查询失败: {e}")
            # 返回 None
            return None


redis_client = RedisClient()
