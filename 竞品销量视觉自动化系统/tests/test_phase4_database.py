"""阶段四验收测试：联合唯一索引 + 分区执行计划 + 账号池 + 会话初始化。

验收标准（来自核心架构开发与验收总览 - 4.2 单元验收标准）：
  测试算子 1：并发 INSERT 两条相同 (virtual_id, record_date) 记录
    → 数据库必须拒绝第二条，抛出 Duplicate entry 异常
  测试算子 2：EXPLAIN SELECT 必须精确指向当前月份物理分区
    → 严禁全表扫描
"""
import sqlite3
import threading
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.db.connection import DatabaseConfig, get_connection, init_schema
from src.db.repository import CompetitorRepository
from src.core.account_pool import AccountPool, Account, AccountStatus, PoolConfig
from src.core.session import SessionManager
from src.core.session_setup import StoreSessionInitializer


# ────────────────────────────────────────────────────────────
# 测试夹具
# ────────────────────────────────────────────────────────────

@pytest.fixture
def db_config(tmp_path):
    """SQLite 内存数据库配置（测试用）。"""
    db_path = tmp_path / "test.db"
    return DatabaseConfig(
        backend="sqlite",
        database=str(db_path),
    )


@pytest.fixture
def repo(db_config):
    """已初始化表结构的仓库实例。"""
    conn = get_connection(db_config)
    init_schema(conn, backend="sqlite")
    conn.close()

    repo = CompetitorRepository(db_config)
    repo.connect()
    yield repo
    repo.close()


# ────────────────────────────────────────────────────────────
# 算子 1：联合唯一索引约束 — 同日重复快照被拒绝
# ────────────────────────────────────────────────────────────

class TestUniqueConstraint:
    """验收算子 1 — (virtual_id, record_date) 联合唯一索引"""

    def test_duplicate_snapshot_rejected(self, repo):
        """同一商品同一天写入两次 → 第二次被拒绝，返回 False。"""
        today = date.today()

        # 先写入店铺和商品
        store_id = repo.find_or_create_store("测试旗舰店")
        virtual_id = repo.find_or_create_product(
            "md5_test_001", store_id, "测试商品A", "abc123",
        )

        # 第一次写入成功
        ok = repo.insert_sales_snapshot(
            virtual_id=virtual_id,
            snapshot_sales=15000,
            record_date=today,
        )
        assert ok, "首次写入应成功"

        # 第二次相同 (virtual_id, record_date) → 被拒绝
        ok2 = repo.insert_sales_snapshot(
            virtual_id=virtual_id,
            snapshot_sales=16000,
            record_date=today,
        )
        assert not ok2, "联合唯一索引应拒绝同日重复写入"

        # 确认只有一条记录
        timeline = repo.get_product_timeline(virtual_id, days=1)
        assert len(timeline) == 1
        assert timeline[0]["snapshot_rolling_sales"] == 15000  # 保持原值

    def test_same_product_different_dates_allowed(self, repo):
        """同一商品不同日期 → 都应成功写入。"""
        store_id = repo.find_or_create_store("测试旗舰店")
        virtual_id = repo.find_or_create_product(
            "md5_test_002", store_id, "测试商品B", "def456",
        )

        for i in range(5):
            ok = repo.insert_sales_snapshot(
                virtual_id=virtual_id,
                snapshot_sales=10000 + i * 100,
                record_date=date.today() - timedelta(days=i),
            )
            assert ok, f"第 {i} 天写入应成功"

        timeline = repo.get_product_timeline(virtual_id, days=30)
        assert len(timeline) == 5

    def test_concurrent_inserts_only_one_succeeds(self, repo):
        """并发写入相同 (virtual_id, record_date) → 仅一条成功。

        模拟凌晨定时任务重复触发场景。
        每个线程打开独立连接到同一个数据库文件。
        """
        today = date.today()
        store_id = repo.find_or_create_store("并发测试店")
        virtual_id = repo.find_or_create_product(
            "md5_concurrent", store_id, "并发商品", "concurrent_hash",
        )

        # 确保 schema 在目标数据库中已存在（fixture 已做，此处确认）
        db_path = repo._config.database

        success_count = [0]
        fail_count = [0]
        errors = []

        def _do_insert():
            try:
                import sqlite3
                # 每个线程打开独立连接到同一个数据库
                conn = sqlite3.connect(db_path)
                conn.execute("PRAGMA journal_mode=WAL")
                try:
                    conn.execute(
                        """INSERT INTO rolling_sales_history
                           (virtual_id, snapshot_rolling_sales, record_date)
                           VALUES (?, ?, ?)""",
                        (virtual_id, 10000, today.isoformat()),
                    )
                    conn.commit()
                    success_count[0] += 1
                except sqlite3.IntegrityError:
                    fail_count[0] += 1
                except Exception as e:
                    errors.append(str(e))
                finally:
                    conn.close()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_do_insert) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"出现非预期错误: {errors}"
        assert success_count[0] == 1, f"应有且仅有一条成功: {success_count[0]}"
        assert fail_count[0] == 4, f"其余 4 条应被拒绝: {fail_count[0]}"


# ────────────────────────────────────────────────────────────
# 算子 2：分区执行计划 — 严禁全表扫描（MySQL 专用）
# ────────────────────────────────────────────────────────────

class TestPartitionPruning:
    """验收算子 2 — EXPLAIN 确认分区裁剪"""

    @pytest.mark.skip(reason="需要 MySQL 连接，CI 环境跳过")
    def test_explain_shows_partition_pruning(self):
        """EXPLAIN SELECT 只扫描当月分区，不全表扫描。"""
        import pymysql

        conn = pymysql.connect(
            host="localhost", port=3306,
            user="root", password="",
            database="competitor_monitor",
        )
        cursor = conn.cursor()

        # 确保使用分区表
        cursor.execute("""
            EXPLAIN SELECT * FROM rolling_sales_history
            WHERE record_date = '2026-05-28'
        """)
        plan = cursor.fetchall()
        for row in plan:
            row_str = str(row)
            # partitions 字段不应为空，且不应包含所有分区
            assert "p2026" in row_str.lower() or "p2025" in row_str.lower(), (
                f"执行计划应命中具体分区: {row_str}"
            )
            assert "p_future" not in row_str or "all" not in row_str.lower(), (
                f"不应全分区扫描: {row_str}"
            )

        cursor.close()
        conn.close()

    def test_sqlite_indexed_query(self, repo):
        """SQLite 下确认索引正常使用（SQLite 无分区，用索引替代验证）。"""
        store_id = repo.find_or_create_store("索引测试店")
        virtual_id = repo.find_or_create_product(
            "md5_index_test", store_id, "索引商品", "idx_hash",
        )

        repo.insert_sales_snapshot(virtual_id, 5000, date.today())

        # SQLite 的 EXPLAIN QUERY PLAN
        conn = repo._conn
        plan = conn.execute(
            """EXPLAIN QUERY PLAN
               SELECT * FROM rolling_sales_history
               WHERE virtual_id = ? AND record_date = ?""",
            (virtual_id, date.today().isoformat()),
        ).fetchall()

        plan_str = str(plan).lower()
        # SQLite 应使用索引而非全表扫描
        assert "scan" not in plan_str or "index" in plan_str or "search" in plan_str, (
            f"应使用索引查找: {plan}"
        )


# ────────────────────────────────────────────────────────────
# 仓库 CRUD 完整测试
# ────────────────────────────────────────────────────────────

class TestRepositoryCRUD:
    """数据访问层基础 CRUD"""

    def test_find_or_create_store_idempotent(self, repo):
        """多次调用返回同一个 store_id。"""
        id1 = repo.find_or_create_store("测试旗舰店")
        id2 = repo.find_or_create_store("测试旗舰店")
        assert id1 == id2

    def test_find_or_create_product(self, repo):
        """商品创建和查找。"""
        store_id = repo.find_or_create_store("测试店")
        vid = repo.find_or_create_product(
            "md5_product_a", store_id,
            "2025春季新款连衣裙女", "phash_abc123",
        )
        assert vid == "md5_product_a"

        # 再次查找返回相同 ID
        vid2 = repo.find_or_create_product(
            "md5_product_a", store_id,
            "不同的标题", "不同的哈希",
        )
        assert vid2 == "md5_product_a"  # 不会覆盖已有记录

    def test_batch_insert_sales(self, repo):
        """批量写入销量快照。"""
        store_id = repo.find_or_create_store("批量测试店")
        today = date.today()

        products = []
        for i in range(10):
            vid = repo.find_or_create_product(
                f"md5_batch_{i}", store_id, f"批量商品_{i}", f"hash_{i}",
            )
            products.append(vid)

        records = [
            {"virtual_id": vid, "snapshot_rolling_sales": 1000 + i * 100,
             "record_date": today, "capture_batch": "batch_001"}
            for i, vid in enumerate(products)
        ]
        count = repo.insert_sales_batch(records)
        assert count == 10

    def test_data_completeness_calculation(self, repo):
        """数据完整率计算。"""
        store_id = repo.find_or_create_store("完整率测试店")
        today = date.today()

        # 创建 5 个商品
        vids = []
        for i in range(5):
            vid = repo.find_or_create_product(
                f"md5_comp_{i}", store_id, f"完整率商品_{i}", f"hash_{i}",
            )
            vids.append(vid)

        # 只写入 3 个商品的快照
        for vid in vids[:3]:
            repo.insert_sales_snapshot(vid, 1000, today)

        completeness = repo.get_data_completeness(today)
        assert completeness["expected"] == 5
        assert completeness["actual"] == 3
        assert completeness["completeness"] == 3.0 / 5.0

    def test_store_daily_aggregate(self, repo):
        """店铺日销量聚合。"""
        store_id = repo.find_or_create_store("聚合测试店")
        today = date.today()

        total = 0
        for i in range(3):
            vid = repo.find_or_create_product(
                f"md5_agg_{i}", store_id, f"聚合商品_{i}", f"hash_{i}",
            )
            sales = 1000 * (i + 1)
            repo.insert_sales_snapshot(vid, sales, today)
            total += sales

        agg = repo.get_store_daily_aggregate(store_id, today)
        assert agg == total


# ────────────────────────────────────────────────────────────
# 多账号资产池测试
# ────────────────────────────────────────────────────────────

class TestAccountPool:
    """账号资产池 — 轮询 + 熔断"""

    def test_rotation_every_n_stores(self):
        """每 N 个店铺轮换账号。"""
        pool = AccountPool(PoolConfig(stores_per_account=3))
        pool.add_account("user_a")
        pool.add_account("user_b")

        # 第一个活跃账号
        acc1 = pool.next_active()
        assert acc1 is not None

        # 完成 3 个店铺后应切到下一个
        for _ in range(3):
            pool.complete_store()
        acc2 = pool.next_active()
        assert acc2 is not None
        assert acc2.username != acc1.username

    def test_abnormal_count_isolation(self):
        """连续 15 次异常文本触发熔断隔离。"""
        pool = AccountPool(PoolConfig(stores_per_account=3, abnormal_threshold=15))
        pool.add_account("doomed_user")
        pool.add_account("spare_user")

        acc = pool.next_active()
        assert acc.username == "doomed_user"

        # 触发 15 次异常
        for _ in range(15):
            pool.report_abnormal_text()

        assert acc.status == AccountStatus.ISOLATED

        # 下次应切换到备用号
        next_acc = pool.next_active()
        assert next_acc.username == "spare_user"

    def test_reset_abnormal_count(self):
        """正常商品文本出现后重置异常计数。"""
        pool = AccountPool(PoolConfig(abnormal_threshold=15))
        pool.add_account("user_x")

        pool.next_active()
        for _ in range(10):
            pool.report_abnormal_text()

        pool.reset_abnormal_count()
        assert pool.current.abnormal_count == 0

    def test_all_isolated_returns_none(self):
        """所有账号被隔离时返回 None。"""
        pool = AccountPool(PoolConfig(abnormal_threshold=5))
        pool.add_account("user_1")
        pool.add_account("user_2")

        pool.next_active()
        for _ in range(5):
            pool.report_abnormal_text()

        acc2 = pool.next_active()
        assert acc2 is not None
        for _ in range(5):
            pool.report_abnormal_text()

        # 两个都被隔离
        acc3 = pool.next_active()
        assert acc3 is None

    def test_isolation_callback(self):
        """隔离时触发告警回调。"""
        alerts = []
        pool = AccountPool(PoolConfig(abnormal_threshold=3))
        pool.add_account("bad_user")
        pool.on_isolated(lambda acc: alerts.append(acc.username))

        pool.next_active()
        for _ in range(3):
            pool.report_abnormal_text()

        assert len(alerts) == 1
        assert alerts[0] == "bad_user"

    def test_pool_stats(self):
        """账号池指标统计。"""
        pool = AccountPool(PoolConfig(abnormal_threshold=3))
        pool.add_account("a1")
        pool.add_account("a2")
        pool.add_account("a3")

        pool.next_active()
        assert pool.stats == {
            "total": 3, "active": 1, "isolated": 0, "idle": 2,
            "current": "a1",
        }


# ────────────────────────────────────────────────────────────
# 会话初始化测试
# ────────────────────────────────────────────────────────────

class TestSessionSetup:
    """店铺会话初始化 — store_id 绑定"""

    def test_initialize_binds_store_id(self, tmp_path):
        """会话初始化完成 store_id 绑定。"""
        from src.core.adb_controller import ADBController

        state_file = tmp_path / "state.json"
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(backend="sqlite", database=str(db_path))

        conn = get_connection(db_config)
        init_schema(conn, backend="sqlite")
        conn.close()

        repo = CompetitorRepository(db_config)
        repo.connect()

        adb = ADBController(adb_path="mock_adb")
        adb._run_adb = MagicMock(return_value="")
        adb.capture_screenshot = MagicMock(return_value=True)
        adb.bezier_swipe = MagicMock()
        session_mgr = SessionManager(state_file=str(state_file))

        init = StoreSessionInitializer(
            adb=adb, repo=repo, session_mgr=session_mgr,
        )

        store_id = init.initialize(store_url_hint="test_store")
        assert store_id.startswith("store_"), f"应生成 store_id: {store_id}"
        assert session_mgr.state.current_store_id == store_id

        repo.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
