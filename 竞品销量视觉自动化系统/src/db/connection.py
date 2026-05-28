"""数据库连接工厂 — 支持 MySQL（生产）和 SQLite（测试）。"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认建表 DDL 路径
_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "sql" / "schema.sql"


class DatabaseConfig:
    def __init__(self, host: str = "localhost", port: int = 3306,
                 user: str = "root", password: str = "",
                 database: str = "competitor_monitor",
                 backend: str = "mysql"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.backend = backend  # "mysql" | "sqlite"


def get_connection(config: DatabaseConfig):
    """获取数据库连接。

    生产环境返回 pymysql 连接，测试环境返回 sqlite3 连接。
    """
    if config.backend == "sqlite":
        import sqlite3
        db_path = Path(config.database)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    else:
        import pymysql
        return pymysql.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database,
            charset="utf8mb4",
        )


def init_schema(conn, backend: str = "mysql"):
    """初始化数据库表结构。

    对于 MySQL，执行 schema.sql 中的创建语句。
    对于 SQLite，执行简化的建表语句。
    """
    if backend == "sqlite":
        _init_sqlite_schema(conn)
    else:
        _init_mysql_schema(conn)


def _init_sqlite_schema(conn):
    """SQLite 简化建表（无分区，用于测试）。"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS competitor_stores (
            store_id    TEXT PRIMARY KEY,
            store_name  TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS competitor_products (
            virtual_id   TEXT PRIMARY KEY,
            store_id     TEXT NOT NULL,
            title        TEXT NOT NULL,
            img_hash     TEXT NOT NULL,
            first_seen_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (store_id) REFERENCES competitor_stores(store_id)
        );

        CREATE TABLE IF NOT EXISTS rolling_sales_history (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            virtual_id              TEXT NOT NULL,
            snapshot_rolling_sales  INTEGER NOT NULL,
            record_date             TEXT NOT NULL,
            created_at              TEXT DEFAULT (datetime('now')),
            capture_batch           TEXT,
            session_label           TEXT DEFAULT 'Normal',
            UNIQUE(virtual_id, record_date)
        );

        CREATE INDEX IF NOT EXISTS idx_sales_virtual_date
            ON rolling_sales_history(virtual_id, record_date);
        CREATE INDEX IF NOT EXISTS idx_sales_record_date
            ON rolling_sales_history(record_date);
        CREATE INDEX IF NOT EXISTS idx_products_store
            ON competitor_products(store_id);
    """)
    conn.commit()


def _init_mysql_schema(conn):
    """执行完整的 MySQL DDL。"""
    if _SCHEMA_PATH.exists():
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        cursor = conn.cursor()
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    cursor.execute(stmt)
                except Exception as e:
                    logger.debug("DDL statement skipped: %s", e)
        conn.commit()
