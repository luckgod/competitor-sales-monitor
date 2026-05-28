"""阶段九验收测试：DPI 映射 + ADB 管道 + 靶点池加载。

验收标准（V5.0 路线图 9.2）：
  T9.1: 坐标映射公式验证
  T9.2: ADB 管道无 fork 开销
  T9.3: targets.yaml 随机打散
  T9.4: Tier 1 优先
"""
import tempfile
from pathlib import Path

import pytest

from src.core.dpi_mapper import DPIMapper, DPIConfig
from src.core.target_loader import TargetLoader, StoreTarget


# ────────────────────────────────────────────────────────────
# T9.1: DPI 坐标映射
# ────────────────────────────────────────────────────────────

class TestDPIMapping:
    """T9.1 — 跨层坐标转换矩阵"""

    def test_pixel_to_abs_corner_cases(self):
        """边界映射: (0,0) → (0,0), (1080,1920) → (max_x, max_y)。"""
        mapper = DPIMapper(DPIConfig(
            abs_max_x=32767, abs_max_y=32767,
            screen_width=1080, screen_height=1920,
        ))

        assert mapper.to_abs(0, 0) == (0, 0)
        assert mapper.to_abs(1080, 1920) == (32767, 32767)

    def test_pixel_to_abs_center(self):
        """中心点映射。"""
        mapper = DPIMapper(DPIConfig(
            abs_max_x=32767, abs_max_y=32767,
            screen_width=1080, screen_height=1920,
        ))

        x, y = mapper.to_abs(540, 960)
        assert abs(x - 16383) < 2  # 接近 32767/2
        assert abs(y - 16383) < 2

    def test_abs_to_pixel_reversible(self):
        """往返映射可逆。"""
        mapper = DPIMapper(DPIConfig(
            abs_max_x=32767, abs_max_y=32767,
            screen_width=1080, screen_height=1920,
        ))

        x_abs, y_abs = mapper.to_abs(540, 960)
        x_px, y_px = mapper.to_pixel(x_abs, y_abs)
        assert abs(x_px - 540) < 2
        assert abs(y_px - 960) < 2

    def test_getevent_parsing(self):
        """getevent -p 输出解析。"""
        sample_output = """
          ABS_MT_POSITION_X     : value 0, min 0, max 32767
          ABS_MT_POSITION_Y     : value 0, min 0, max 32767
        """
        max_x = DPIMapper._parse_abs_max(sample_output, "ABS_MT_POSITION_X")
        max_y = DPIMapper._parse_abs_max(sample_output, "ABS_MT_POSITION_Y")
        assert max_x == 32767
        assert max_y == 32767

    def test_different_resolution_mapping(self):
        """不同分辨率设备映射正确。"""
        # 720P 手机
        mapper = DPIMapper(DPIConfig(
            abs_max_x=32767, abs_max_y=32767,
            screen_width=720, screen_height=1280,
        ))

        x_abs, y_abs = mapper.to_abs(720, 1280)
        assert x_abs == 32767
        assert y_abs == 32767


# ────────────────────────────────────────────────────────────
# T9.3 + T9.4: 靶点池加载与打散
# ────────────────────────────────────────────────────────────

class TestTargetLoader:
    """T9.3/T9.4 — targets.yaml 随机打散 + Tier 优先"""

    @pytest.fixture
    def yaml_path(self, tmp_path):
        """创建测试用 targets.yaml。"""
        content = """
store_pool:
  tier_1:
    - store_name: "店铺A"
      keywords: ["关键词A1", "关键词A2"]
    - store_name: "店铺B"
      keywords: ["关键词B1"]
    - store_name: "店铺C"
      keywords: ["关键词C1"]
  tier_2:
    - store_name: "店铺D"
      keywords: ["关键词D1"]
    - store_name: "店铺E"
      keywords: ["关键词E1"]
"""
        p = tmp_path / "targets.yaml"
        p.write_text(content, encoding="utf-8")
        return str(p)

    def test_load_targets(self, yaml_path):
        """靶点池加载正确。"""
        loader = TargetLoader(yaml_path)
        targets = loader.load()

        assert len(targets) == 5
        assert loader.tier1_count == 3
        assert loader.tier2_count == 2

    def test_shuffled_tier1_first(self, yaml_path):
        """T9.4: Tier 1 排在 Tier 2 前面。"""
        loader = TargetLoader(yaml_path)
        loader.load()

        result = loader.shuffled(tier1_first=True)

        # 前 3 个应为 Tier 1
        for i in range(3):
            assert result[i].tier == 1
        # 后 2 个应为 Tier 2
        for i in range(3, 5):
            assert result[i].tier == 2

    def test_shuffled_random_order(self, yaml_path):
        """T9.3: 多次调用 shuffled 顺序不同（probabilistic）。"""
        loader = TargetLoader(yaml_path)
        loader.load()

        orders = []
        for _ in range(10):
            result = loader.shuffled()
            orders.append(tuple(t.store_name for t in result))

        # 至少有一次不同的排列（概率极高）
        unique = len(set(orders))
        assert unique >= 2, f"10 次打散应至少产生 2 种不同排列: {unique}"

    def test_get_tier1_keywords(self, yaml_path):
        """关键词查询正确。"""
        loader = TargetLoader(yaml_path)
        loader.load()

        keywords = loader.get_tier1_keywords("店铺A")
        assert keywords == ["关键词A1", "关键词A2"]

    def test_empty_file_returns_empty(self, tmp_path):
        """配置文件为空时返回空列表。"""
        p = tmp_path / "empty.yaml"
        p.write_text("{}", encoding="utf-8")

        loader = TargetLoader(str(p))
        targets = loader.load()
        assert targets == []

    def test_missing_file_handled(self):
        """配置文件不存在时不抛异常。"""
        loader = TargetLoader("nonexistent.yaml")
        targets = loader.load()
        assert targets == []

    def test_keywords_default_empty(self, yaml_path):
        """未指定 keywords 时为空列表。"""
        content = """
store_pool:
  tier_1:
    - store_name: "无关键词店"
"""
        p = Path(yaml_path).parent / "no_kw.yaml"
        p.write_text(content, encoding="utf-8")

        loader = TargetLoader(str(p))
        targets = loader.load()
        assert targets[0].keywords == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
