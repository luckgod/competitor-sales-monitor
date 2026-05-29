import sqlite3
c=sqlite3.connect('data/topology.db')
rows=c.execute('SELECT virtual_id, title, img_hash, first_seen FROM competitor_products').fetchall()
print(f"店铺: 朱公子上岸教育")
print(f"商品总数: {len(rows)}")
print("=" * 60)
for i,(vid,title,ph,dt) in enumerate(rows):
    print(f"{i+1:2d}. vid={vid}  phash={ph[:12]}  {dt}")
c.close()
