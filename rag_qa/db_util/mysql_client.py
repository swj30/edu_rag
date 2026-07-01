# 本模块负责mysql数据表的操作
import pymysql
import pandas as pd
from base.config import config as conf
from base.logger import logger


class MySQLClient:
    def __init__(self):
        try:
            # 连接 MySQL 数据库
            self.connection = pymysql.connect(
                host=conf.MYSQL_HOST,
                user=conf.MYSQL_USER,
                password=conf.MYSQL_PASSWORD,
                database=conf.MYSQL_DATABASE,
            )
            # 创建游标
            self.cursor = self.connection.cursor()
            # 记录连接成功
            logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            # 记录连接失败
            logger.error(f"MySQL 连接失败: {e}")
            raise

    # 插入数据
    def insert_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            for _, row in data.iterrows():
                insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                self.cursor.execute(
                    insert_query, (row["学科名称"], row["问题"], row["答案"])
                )
            self.connection.commit()
            logger.info("数据插入成功")
        except Exception as e:
            logger.error(f"数据插入失败: {e}")
            self.connection.rollback()
            raise

    # 查询所有问题
    def fetch_questions(self):
        # 获取所有问题
        try:
            # 执行查询
            self.cursor.execute("SELECT question FROM jpkb")
            # 获取结果
            results = self.cursor.fetchall()
            # 记录获取成功
            logger.info("成功获取问题")
            # 返回结果
            return results
        except pymysql.MySQLError as e:
            # 记录查询失败
            logger.error(f"查询失败: {e}")
            # 返回空列表
            return []

    # 查询指定问题的答案
    def fetch_answer(self, question):
        # 获取指定问题的答案
        try:
            # 执行查询
            self.cursor.execute(
                "SELECT answer FROM jpkb WHERE question=%s", (question,)
            )
            # 获取结果
            result = self.cursor.fetchone()
            # 返回答案或 None
            return result[0] if result else None
        except pymysql.MySQLError as e:
            # 记录答案获取失败
            logger.error(f"答案获取失败: {e}")
            # 返回 None
            return None

    # 获取会话历史--查询最近的5轮对话
    def fetch_recent_history(self, session_id: str) -> list:
        """获取最近5轮对话历史"""
        try:
            # 执行 SQL 查询，获取最近 5 轮对话
            self.cursor.execute(
                """
                    SELECT question, answer
                    FROM conversations
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """,
                (session_id, 5),
            )
            # 将查询结果转换为字典列表
            history = [
                {"question": row[0], "answer": row[1]} for row in self.cursor.fetchall()
            ]
            # 反转结果，按时间正序返回
            return history[::-1]
        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []

    # 更新会话历史
    def update_session_history(
        self, session_id: str, question: str, answer: str
    ) -> list:
        """更新会话历史到MySQL，保留最近5轮对话"""
        try:
            # 插入新的对话记录
            self.cursor.execute(
                """
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """,
                (session_id, question, answer),
            )
            # 获取更新后的对话历史
            history = self.fetch_recent_history(session_id)
            # 删除超出 5 轮的旧记录
            self.cursor.execute(
                """
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS sub
                )
            """,
                (session_id, session_id, 5),
            )
            # 提交事务
            self.connection.commit()
            # 记录更新成功的日志
            logger.info(f"会话 {session_id} 历史更新成功")
            # 返回更新后的历史
            return history
        except pymysql.MySQLError as e:
            # 记录数据库操作失败的错误日志
            logger.error(f"更新会话历史失败: {e}")
            # 回滚事务
            mysql_client.connection.rollback()
            # 抛出异常
            raise
        except Exception as e:
            # 记录意外错误的日志
            logger.error(f"更新会话历史意外错误: {e}")
            # 回滚事务
            mysql_client.connection.rollback()
            # 抛出异常
            raise

    # 清除会话历史
    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        try:
            # 删除指定 session_id 的所有对话记录
            self.cursor.execute(
                """
                DELETE FROM conversations
                WHERE session_id = %s
            """,
                (session_id,),
            )
            # 提交事务
            self.connection.commit()
            # 记录清除成功的日志
            logger.info(f"会话 {session_id} 历史已清除")
            # 返回 True 表示成功
            return True
        except pymysql.MySQLError as e:
            # 记录清除失败的错误日志
            logger.error(f"清除会话历史失败: {e}")
            # 回滚事务
            self.connection.rollback()
            # 返回 False 表示失败
            return False

    # 关闭数据库连接
    def close(self):
        try:
            # 关闭连接
            self.connection.close()
            # 记录关闭成功
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            logger.error(f"关闭连接失败: {e}")


mysql_client = MySQLClient()

if __name__ == "__main__":
    mysql_client.insert_data("../../data/JP学科知识问答.csv")
