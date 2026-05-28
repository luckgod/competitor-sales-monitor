"""微观订单流水数据访问层 — INSERT IGNORE 批量写入。

设计文档 5.4：
- 联合唯一索引 (virtual_id, buyer_mask, sku_name, order_date)
- INSERT IGNORE 去重
"""
import logging
from datetime import date
from typing import Optional

from .connection import DatabaseConfig, get_connection

logger = logging.getLogger(__name__)


class OrderRepository:
    """实时订单流水 CRUD。"""

    def __init__(self, config: DatabaseConfig):
        self._config = config
        self._conn = None

    def connect(self) -> None:
        self._conn = get_connection(self._config)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def insert_order(self, virtual_id: str, buyer_mask: str,
                     sku_name: str, is_repeat: bool = False,
                     order_date: str = "",
                     capture_batch_id: str = "") -> bool:
        """单条写入（自动去重）。

        Returns:
            True 表示新记录写入成功，False 表示重复被跳过。
        """
        try:
            if self._config.backend == "sqlite":
                sql = """INSERT OR IGNORE INTO competitor_realtime_orders
                         (virtual_id, buyer_mask, sku_name, is_repeat, order_date, capture_batch_id)
                         VALUES (?, ?, ?, ?, ?, ?)"""
            else:
                sql = """INSERT IGNORE INTO competitor_realtime_orders
                         (virtual_id, buyer_mask, sku_name, is_repeat, order_date, capture_batch_id)
                         VALUES (%s, %s, %s, %s, %s, %s)"""

            cursor = self._conn.cursor()
            cursor.execute(sql, (
                virtual_id, buyer_mask, sku_name,
                1 if is_repeat else 0, order_date, capture_batch_id,
            ))
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            msg = str(e).lower()
            if "duplicate" in msg or "unique" in msg or "UNIQUE constraint" in msg:
                return False
            logger.exception("订单写入失败")
            return False

    def insert_batch(self, orders: list[dict]) -> int:
        """批量写入订单（单条 SQL 批量提交）。

        Args:
            orders: [{"virtual_id": ..., "buyer_mask": ..., "sku_name": ...,
                      "is_repeat": ..., "order_date": ..., "capture_batch_id": ...}, ...]

        Returns:
            成功写入条数
        """
        if not orders:
            return 0

        try:
            if self._config.backend == "sqlite":
                return self._batch_sqlite(orders)
            else:
                return self._batch_mysql(orders)
        except Exception:
            logger.exception("批量订单写入失败")
            return 0

    def _batch_sqlite(self, orders: list[dict]) -> int:
        count = 0
        for o in orders:
            if self.insert_order(**{
                "virtual_id": o["virtual_id"],
                "buyer_mask": o["buyer_mask"],
                "sku_name": o["sku_name"],
                "is_repeat": o.get("is_repeat", False),
                "order_date": o.get("order_date", ""),
                "capture_batch_id": o.get("capture_batch_id", ""),
            }):
                count += 1
        return count

    def _batch_mysql(self, orders: list[dict]) -> int:
        import pymysql
        cursor = self._conn.cursor()

        sql = """INSERT IGNORE INTO competitor_realtime_orders
                 (virtual_id, buyer_mask, sku_name, is_repeat, order_date, capture_batch_id)
                 VALUES (%s, %s, %s, %s, %s, %s)"""

        values = [
            (o["virtual_id"], o["buyer_mask"], o["sku_name"],
             1 if o.get("is_repeat") else 0, o["order_date"],
             o.get("capture_batch_id", ""))
            for o in orders
        ]

        try:
            cursor.executemany(sql, values)
            self._conn.commit()
            return cursor.rowcount
        except Exception:
            logger.exception("MySQL 批量写入失败")
            return 0

    # ── 查询 ──────────────────────────────────────────────────

    def get_product_orders(self, virtual_id: str,
                            days: int = 7) -> list[dict]:
        """获取某商品近 N 天订单。"""
        rows = self._fetch_all(
            """SELECT buyer_mask, sku_name, is_repeat, order_date, capture_time
               FROM competitor_realtime_orders
               WHERE virtual_id = ?
               ORDER BY order_date DESC, capture_time DESC
               LIMIT 1000""",
            (virtual_id,),
        )
        return [
            {"buyer_mask": r[0], "sku_name": r[1], "is_repeat": bool(r[2]),
             "order_date": r[3], "capture_time": r[4]}
            for r in rows
        ]

    def get_daily_order_count(self, virtual_id: str,
                               target_date: str) -> int:
        """某商品某日订单数。"""
        row = self._fetch_one(
            """SELECT COUNT(*) FROM competitor_realtime_orders
               WHERE virtual_id = ? AND order_date = ?""",
            (virtual_id, target_date),
        )
        return row[0] if row else 0

    # ── 内部 ──────────────────────────────────────────────────

    def _fetch_one(self, sql: str, params: tuple = ()):
        if self._config.backend == "sqlite":
            sql = sql.replace("%s", "?")
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchone()

    def _fetch_all(self, sql: str, params: tuple = ()) -> list:
        if self._config.backend == "sqlite":
            sql = sql.replace("%s", "?")
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()
