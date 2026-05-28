"""阶段八验收测试：实时订单流水表 + 微批次双缓冲。

验收标准（V5.0 路线图 8.2）：
  T8.1: 同一订单 INSERT 两次 → 第二条被 IGNORE
  T8.2: 写入 29 条 + 5s 超时 → 自动刷盘
  T8.3: 写入 30 条 → 立即刷盘
  T8.4: emergency_flush → 残留数据不丢失
  T8.5: 双缓冲交替 → 写入期间不阻塞
"""
import time
from datetime import date
from pathlib import Path

import pytest

from src.db.connection import DatabaseConfig, get_connection
from src.db.order_repository import OrderRepository
from src.pipeline.micro_batch import MicroBatchBuffer


# ────────────────────────────────────────────────────────────
# 测试夹具
# ────────────────────────────────────────────────────────────

def _init_sqlite_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS competitor_realtime_orders (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            virtual_id      TEXT NOT NULL,
            buyer_mask      TEXT NOT NULL,
            sku_name        TEXT NOT NULL,
            is_repeat       INTEGER DEFAULT 0,
            order_date      TEXT NOT NULL,
            capture_batch_id TEXT DEFAULT NULL,
            capture_time    TEXT DEFAULT (datetime('now')),
            UNIQUE(virtual_id, buyer_mask, sku_name, order_date)
        );
    """)
    conn.commit()


@pytest.fixture
def db_config(tmp_path):
    db_path = tmp_path / "test_orders.db"
    return DatabaseConfig(backend="sqlite", database=str(db_path))


@pytest.fixture
def repo(db_config):
    conn = get_connection(db_config)
    _init_sqlite_schema(conn)
    conn.close()

    repo = OrderRepository(db_config)
    repo.connect()
    yield repo
    repo.close()


def _make_order(vid="md5_prod_001", buyer="不**", sku="测试课", repeat=False, date_str=None):
    return {
        "virtual_id": vid,
        "buyer_mask": buyer,
        "sku_name": sku,
        "is_repeat": repeat,
        "order_date": date_str or date.today().isoformat(),
        "capture_batch_id": "batch_test",
    }


# ────────────────────────────────────────────────────────────
# T8.1: 联合唯一索引去重
# ────────────────────────────────────────────────────────────

class TestOrderDedup:
    """T8.1 — INSERT IGNORE 去重"""

    def test_duplicate_order_rejected(self, repo):
        """同一订单写入两次 → 第二条被静默丢弃。"""
        o = _make_order()

        ok1 = repo.insert_order(**o)
        assert ok1, "首次写入应成功"

        ok2 = repo.insert_order(**o)
        assert not ok2, "重复订单应被拒绝"

    def test_different_sku_allowed(self, repo):
        """不同 SKU 的订单可正常写入。"""
        o1 = _make_order(sku="SKU_A")
        o2 = _make_order(sku="SKU_B")

        assert repo.insert_order(**o1)
        assert repo.insert_order(**o2)

    def test_different_date_allowed(self, repo):
        """不同日期的同 SKU 可写入。"""
        o1 = _make_order(date_str="2026-05-28")
        o2 = _make_order(date_str="2026-05-29")

        assert repo.insert_order(**o1)
        assert repo.insert_order(**o2)

    def test_batch_insert_with_duplicates(self, repo):
        """批量写入含重复项 → 重复项被跳过。"""
        orders = [
            _make_order(buyer="A**"),
            _make_order(buyer="B**"),
            _make_order(buyer="A**"),  # 重复
        ]
        count = repo.insert_batch(orders)
        assert count == 2  # A, B


# ────────────────────────────────────────────────────────────
# T8.2 + T8.3: 微批次双缓冲区
# ────────────────────────────────────────────────────────────

class TestMicroBatchBuffer:
    """T8.2/T8.3/T8.4/T8.5 — 微批次刷盘"""

    def test_flush_at_capacity_30(self):
        """T8.3: 写入 30 条 → 立即刷盘。"""
        flushed = []

        buf = MicroBatchBuffer(
            batch_size=30,
            flush_callback=lambda orders: flushed.extend(orders) or len(orders),
        )

        for i in range(30):
            buf.insert({"id": i})

        assert len(flushed) == 30
        assert buf.pending_count == 0

    def test_timer_flush_after_5s(self):
        """T8.2: 写入 29 条 → 5s 后自动刷盘。"""
        flushed = []

        buf = MicroBatchBuffer(
            batch_size=30,
            flush_interval=0.3,  # 加速测试
            flush_callback=lambda orders: flushed.extend(orders) or len(orders),
        )
        buf.start_timer()

        for i in range(10):
            buf.insert({"id": i})

        # 等待定时器触发
        time.sleep(0.8)

        buf.stop_timer()
        assert len(flushed) > 0, "定时器应触发自动刷盘"

    def test_emergency_flush_saves_data(self):
        """T8.4: emergency_flush 抢救缓冲区残留。"""
        flushed = []

        buf = MicroBatchBuffer(
            batch_size=30,
            flush_callback=lambda orders: flushed.extend(orders) or len(orders),
        )

        for i in range(15):
            buf.insert({"id": i})

        # 未达到 30 条阈值
        assert buf.pending_count == 15

        # 紧急刷盘
        count = buf.emergency_flush()
        assert count == 15
        assert buf.pending_count == 0
        assert buf.stats["total_emergency_flushes"] == 1

    def test_double_buffer_non_blocking(self):
        """T8.5: 双缓冲交替 → 刷盘期间不阻塞写入。"""
        import threading

        flush_log = []
        insert_log = []

        def flush_cb(orders):
            flush_log.append(len(orders))
            # 模拟慢速 I/O
            time.sleep(0.1)
            return len(orders)

        buf = MicroBatchBuffer(
            batch_size=5,  # 小批次加速测试
            flush_callback=flush_cb,
        )

        errors = []

        def producer():
            try:
                for i in range(20):
                    buf.insert({"id": i})
                    insert_log.append(i)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        t = threading.Thread(target=producer)
        t.start()
        t.join(timeout=5)

        assert len(errors) == 0, f"写入不应抛异常: {errors}"
        total_flushed = sum(flush_log)
        assert total_flushed >= 10, f"至少应有部分刷盘: {total_flushed}"

    def test_stats_accuracy(self):
        """指标统计正确。"""
        flushed = []

        buf = MicroBatchBuffer(
            batch_size=5,
            flush_callback=lambda orders: flushed.extend(orders) or len(orders),
        )

        for i in range(5):
            buf.insert({"id": i})

        stats = buf.stats
        assert stats["total_flushes"] == 1
        assert stats["total_inserted"] == 5
        assert stats["pending"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
