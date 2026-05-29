"""数据访问层 — 竞品店铺/单品/快照 CRUD。

约束：
- (virtual_id, record_date) 联合唯一索引 → 同日重复快照自动拒绝
- 按月分区 → 写入性能恒定 O(log N_month)
"""
import logging
from datetime import date, datetime
from typing import Optional

from .connection import DatabaseConfig, get_connection

logger = logging.getLogger(__name__)


class CompetitorRepository:
    """竞品数据库访问层。

    封装店铺、单品、销量快照的 CRUD 操作。
    支持 MySQL 和 SQLite 双后端。
    """

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

    # ── 店铺 ──────────────────────────────────────────────────

    def find_or_create_store(self, store_name: str) -> str:
        """根据店铺名称查找或创建 store_id。"""
        import uuid

        # 查找已有
        row = self._fetch_one(
            "SELECT store_id FROM competitor_stores WHERE store_name = ?",
            (store_name,),
        )
        if row:
            return row[0]

        store_id = "store_" + uuid.uuid4().hex[:12]
        self._execute(
            "INSERT INTO competitor_stores (store_id, store_name) VALUES (?, ?)",
            (store_id, store_name),
        )
        return store_id

    # ── 单品 ──────────────────────────────────────────────────

    def find_or_create_product(self, virtual_id: str, store_id: str,
                                title: str, img_hash: str) -> str:
        """查找或创建商品记录，返回 virtual_id。"""
        row = self._fetch_one(
            "SELECT virtual_id FROM competitor_products WHERE virtual_id = ?",
            (virtual_id,),
        )
        if row:
            return row[0]

        self._execute(
            """INSERT INTO competitor_products (virtual_id, store_id, title, img_hash)
               VALUES (?, ?, ?, ?)""",
            (virtual_id, store_id, title, img_hash),
        )
        return virtual_id

    # ── 销量快照 ──────────────────────────────────────────────

    def insert_sales_snapshot(self, virtual_id: str,
                               snapshot_sales: int,
                               record_date: date,
                               capture_batch: str = "",
                               session_label: str = "Normal") -> bool:
        """写入当日销量快照。

        若 (virtual_id, record_date) 已存在则静默跳过（联合唯一索引约束）。
        若全局熔断已触发则拒绝写入。

        Returns:
            True 表示写入成功，False 表示重复被跳过或熔断拒绝。
        """
        from .killswitch import is_killed
        if is_killed():
            logger.warning("熔断已触发，拒绝写入: %s / %s", virtual_id, record_date)
            return False
        try:
            self._execute(
                """INSERT INTO rolling_sales_history
                   (virtual_id, snapshot_rolling_sales, record_date,
                    capture_batch, session_label)
                   VALUES (?, ?, ?, ?, ?)""",
                (virtual_id, snapshot_sales,
                 record_date.isoformat() if isinstance(record_date, date) else record_date,
                 capture_batch, session_label),
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            if "duplicate" in msg or "unique" in msg or "UNIQUE constraint" in msg:
                logger.debug("快照已存在，跳过: %s / %s", virtual_id, record_date)
                return False
            raise

    def insert_sales_batch(self, records: list[dict]) -> int:
        """批量写入销量快照。

        Args:
            records: [{"virtual_id": ..., "snapshot_rolling_sales": ...,
                       "record_date": ..., "capture_batch": ..., "session_label": ...}, ...]

        Returns:
            成功写入的条数（跳过重复的）。
        """
        count = 0
        for r in records:
            if self.insert_sales_snapshot(
                virtual_id=r["virtual_id"],
                snapshot_sales=r["snapshot_rolling_sales"],
                record_date=r["record_date"],
                capture_batch=r.get("capture_batch", ""),
                session_label=r.get("session_label", "Normal"),
            ):
                count += 1
        return count

    # ── 查询 ──────────────────────────────────────────────────

    def get_product_timeline(self, virtual_id: str,
                              days: int = 30) -> list[dict]:
        """获取某商品近 N 天的销量快照时间序列。"""
        rows = self._fetch_all(
            """SELECT record_date, snapshot_rolling_sales, session_label
               FROM rolling_sales_history
               WHERE virtual_id = ?
               ORDER BY record_date DESC
               LIMIT ?""",
            (virtual_id, days),
        )
        return [
            {"record_date": r[0], "snapshot_rolling_sales": r[1],
             "session_label": r[2]}
            for r in rows
        ]

    def get_store_daily_aggregate(self, store_id: str,
                                   target_date: date) -> int:
        """获取某店铺某日所有单品销量总和。"""
        row = self._fetch_one(
            """SELECT COALESCE(SUM(h.snapshot_rolling_sales), 0)
               FROM rolling_sales_history h
               JOIN competitor_products p ON h.virtual_id = p.virtual_id
               WHERE p.store_id = ? AND h.record_date = ?""",
            (store_id, target_date.isoformat() if isinstance(target_date, date) else target_date),
        )
        return row[0] if row else 0

    def get_data_completeness(self, target_date: date) -> dict:
        """获取指定日期的数据完整率。

        Returns:
            {"expected": int, "actual": int, "completeness": float}
        """
        expected = self._fetch_one(
            "SELECT COUNT(*) FROM competitor_products"
        )
        actual = self._fetch_one(
            """SELECT COUNT(DISTINCT virtual_id)
               FROM rolling_sales_history
               WHERE record_date = ?""",
            (target_date.isoformat(),),
        )
        exp = expected[0] if expected else 0
        act = actual[0] if actual else 0
        return {
            "expected": exp,
            "actual": act,
            "completeness": act / max(exp, 1),
        }

    # ── 内部工具 ──────────────────────────────────────────────

    def _execute(self, sql: str, params: tuple = ()) -> None:
        if self._config.backend == "sqlite":
            sql = sql.replace("%s", "?").replace("NOW()", "datetime('now')")
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        self._conn.commit()

    def _fetch_one(self, sql: str, params: tuple = ()):
        if self._config.backend == "sqlite":
            sql = sql.replace("%s", "?").replace("NOW()", "datetime('now')")
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchone()

    def _fetch_all(self, sql: str, params: tuple = ()) -> list:
        if self._config.backend == "sqlite":
            sql = sql.replace("%s", "?").replace("NOW()", "datetime('now')")
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()
