-- V5.0 新增：微观订单流水表 DDL
-- MySQL 8.0+

USE competitor_monitor;

CREATE TABLE IF NOT EXISTS competitor_realtime_orders (
    id              BIGINT          AUTO_INCREMENT,
    virtual_id      VARCHAR(64)     NOT NULL,
    buyer_mask      VARCHAR(32)     NOT NULL,
    sku_name        VARCHAR(128)    NOT NULL,
    is_repeat       BOOLEAN         DEFAULT FALSE,
    order_date      DATE            NOT NULL,
    capture_batch_id VARCHAR(16)    DEFAULT NULL,
    capture_time    TIMESTAMP       DEFAULT NOW(),
    PRIMARY KEY (id),
    INDEX idx_virtual_id (virtual_id),
    INDEX idx_order_date (order_date),
    UNIQUE KEY uk_order_event_dedup (virtual_id, buyer_mask, sku_name, order_date)
) ENGINE=InnoDB
PARTITION BY RANGE (TO_DAYS(order_date)) (
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
