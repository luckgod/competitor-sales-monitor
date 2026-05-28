-- 竞品销量纯视觉自动化统计系统 — 数据库 DDL
-- MySQL 8.0+

CREATE DATABASE IF NOT EXISTS competitor_monitor
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE competitor_monitor;

-- 4.1 竞品店铺表
CREATE TABLE competitor_stores (
    store_id    VARCHAR(64)     NOT NULL,
    store_name  VARCHAR(100)    NOT NULL,
    created_at  TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (store_id)
) ENGINE=InnoDB;

-- 4.2 竞品单品表
CREATE TABLE competitor_products (
    virtual_id   VARCHAR(64)    NOT NULL,
    store_id     VARCHAR(64)    NOT NULL,
    title        VARCHAR(255)   NOT NULL,
    img_hash     VARCHAR(64)    NOT NULL,
    first_seen_at TIMESTAMP     DEFAULT NOW(),
    PRIMARY KEY (virtual_id),
    INDEX idx_store (store_id),
    CONSTRAINT fk_product_store FOREIGN KEY (store_id)
        REFERENCES competitor_stores(store_id)
) ENGINE=InnoDB;

-- 4.3 滚动销量历史快照表（按月分区）
CREATE TABLE rolling_sales_history (
    id                      BIGINT          AUTO_INCREMENT,
    virtual_id              VARCHAR(64)     NOT NULL,
    snapshot_rolling_sales  INT             NOT NULL,
    record_date             DATE            NOT NULL,
    created_at              TIMESTAMP       DEFAULT NOW(),
    capture_batch           VARCHAR(16)     DEFAULT NULL COMMENT '采集批次标识',
    session_label           VARCHAR(32)     DEFAULT 'Normal' COMMENT 'Normal | Replenish_Snapshot',
    PRIMARY KEY (id, record_date),
    UNIQUE KEY uk_virtual_date (virtual_id, record_date),
    INDEX idx_record_date (record_date)
) ENGINE=InnoDB
PARTITION BY RANGE (TO_DAYS(record_date)) (
    PARTITION p2025_01 VALUES LESS THAN (TO_DAYS('2025-02-01')),
    PARTITION p2025_02 VALUES LESS THAN (TO_DAYS('2025-03-01')),
    PARTITION p2025_03 VALUES LESS THAN (TO_DAYS('2025-04-01')),
    PARTITION p2025_04 VALUES LESS THAN (TO_DAYS('2025-05-01')),
    PARTITION p2025_05 VALUES LESS THAN (TO_DAYS('2025-06-01')),
    PARTITION p2025_06 VALUES LESS THAN (TO_DAYS('2025-07-01')),
    PARTITION p2025_07 VALUES LESS THAN (TO_DAYS('2025-08-01')),
    PARTITION p2025_08 VALUES LESS THAN (TO_DAYS('2025-09-01')),
    PARTITION p2025_09 VALUES LESS THAN (TO_DAYS('2025-10-01')),
    PARTITION p2025_10 VALUES LESS THAN (TO_DAYS('2025-11-01')),
    PARTITION p2025_11 VALUES LESS THAN (TO_DAYS('2025-12-01')),
    PARTITION p2025_12 VALUES LESS THAN (TO_DAYS('2026-01-01')),
    PARTITION p_future    VALUES LESS THAN MAXVALUE
);
