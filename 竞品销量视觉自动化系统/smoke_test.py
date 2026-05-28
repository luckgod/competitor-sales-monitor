"""实战冒烟测试 — 无外部依赖的核心模块验证"""
import numpy as np
from datetime import date, timedelta

print("=== 1. 销量提取器 Regex 主闸门 ===")
from src.ocr.sales_extractor import SalesExtractor
ext = SalesExtractor()

tests = [
    ("月销 1.5万+", 15000),
    ("已售 100件", 100),
    ("4200+人付款", 4200),
    ("付款 888", 888),
    ("销量 5000+", 5000),
    ("5000", 5000),
]
all_ok = True
for text, expected in tests:
    result = ext.extract(text)
    ok = result == expected
    if not ok:
        all_ok = False
    print(f"  '{text}' -> {result} {'OK' if ok else f'FAIL(expected {expected})'}")

print(f"  regex命中: {ext.stats['regex_hits']}/{len(tests)}")

print("\n=== 2. 懒加载无效帧过滤 ===")
from src.consumer.filter import LazyLoadFilter
f = LazyLoadFilter()
white = np.full((200, 200, 3), 255, dtype=np.uint8)
texture = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
print(f"  纯白卡 should_invalid: {f.is_invalid({'image': white, 'sales': white})}")
print(f"  纹理卡 should_pass:    {not f.is_invalid({'image': texture, 'sales': texture})}")

print("\n=== 3. pHash 去重引擎 ===")
from src.consumer.dedup import DedupEngine
dedup = DedupEngine(window_size=30)
for i in range(5):
    dup = dedup.is_duplicate("爆款连衣裙夏季新款", texture.copy())
    print(f"  第{i+1}次相同商品: duplicate={dup}")
print(f"  stats: miss={dedup.stats['miss_count']}, hit={dedup.stats['hit_count']}")

print("\n=== 4. 弹窗遮挡检测 ===")
from src.ui_guard.overlay_detector import OverlayDetector
od = OverlayDetector()
for i in range(5):
    od.check(["低电量", "请充电", "确定"], card_regions_valid=False)
print(f"  连续5帧低电量弹窗: blocked={od.is_blocked}")

print("\n=== 5. UI改版熔断锁 ===")
from src.ui_guard.layout_guard import LayoutGuard
guard = LayoutGuard()
guard.calibrate([{"w": 540, "h": 580}] * 5)
for i in range(19):
    guard.check({"w": 200, "h": 800}, sales_value=100)
print(f"  19帧后 killed={guard.is_killed}")
guard.check({"w": 200, "h": 800}, sales_value=100)
print(f"  第20帧后 killed={guard.is_killed}")

print("\n=== 6. 数据修复线性插值 ===")
from src.core.data_healer import DataHealer
by_date = {date.today() - timedelta(days=2): 1000, date.today(): 2000}
val = DataHealer._linear_interpolate(by_date, date.today() - timedelta(days=1))
print(f"  缺失昨日数据: 插值={val} (预期1500) {'OK' if val == 1500 else 'FAIL'}")

print("\n=== 7. 店铺创建 + 快照写入 (SQLite) ===")
from src.db.connection import DatabaseConfig, get_connection, init_schema
from src.db.repository import CompetitorRepository
import tempfile, os

db_path = os.path.join(tempfile.gettempdir(), "smoke_test.db")
config = DatabaseConfig(backend="sqlite", database=db_path)
conn = get_connection(config)
init_schema(conn, backend="sqlite")
conn.close()

repo = CompetitorRepository(config)
repo.connect()
store_id = repo.find_or_create_store("冒烟测试旗舰店")
print(f"  store_id: {store_id}")
vid = repo.find_or_create_product("md5_smoke_001", store_id, "冒烟商品A", "hash_a1b2")
print(f"  virtual_id: {vid}")
ok = repo.insert_sales_snapshot(vid, 15000, date.today())
print(f"  快照写入: {'OK' if ok else 'DUPLICATE'}")
dup = repo.insert_sales_snapshot(vid, 16000, date.today())
print(f"  重复写入: {'BLOCKED' if not dup else 'ERROR(should block)'}")
completeness = repo.get_data_completeness(date.today())
print(f"  数据完整率: {completeness}")
repo.close()
os.remove(db_path)

print("\n=== 8. 多账号资产池 ===")
from src.core.account_pool import AccountPool, PoolConfig
pool = AccountPool(PoolConfig(stores_per_account=3, abnormal_threshold=5))
pool.add_account("user_a")
pool.add_account("user_b")
acc = pool.next_active()
print(f"  当前账号: {acc.username}")
for _ in range(3):
    pool.complete_store()
acc2 = pool.next_active()
print(f"  3店后切换: {acc2.username}")

print(f"\n冒烟测试{'全部通过' if all_ok else '有失败项'}!")
