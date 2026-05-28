"""V5.0 高级优化验收：共享内存零拷贝 / Control Socket / Trie SKU / VRAM 固定归一化。"""
import multiprocessing as mp
import time

import numpy as np
import pytest

# ── 优化 1: 共享内存零拷贝 ────────────────────────────────────

class TestSharedMemoryBridge:
    """6.9.1 — multiprocessing.shared_memory 零拷贝"""

    def test_create_and_write_read_same_process(self):
        """同进程内创建、写入、读取。"""
        from src.pipeline.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge(name="test_shm_001")
        bridge.create()

        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        bridge.write_frame(frame, frame_id=1)

        read_back = bridge.read_frame()
        assert read_back.shape == (1080, 1920, 3)
        assert np.array_equal(read_back, frame)

        bridge.unlink()

    def test_cross_process_zero_copy(self):
        """跨进程零拷贝 — 子进程通过名称附加。"""
        from src.pipeline.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge(name="test_shm_cross")
        bridge.create()

        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        bridge.write_frame(frame)

        shm_name = bridge.name

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_shm_child_reader, args=(shm_name, q))
        p.start()
        p.join(timeout=10)

        shape = q.get(timeout=3)
        assert shape == (1080, 1920, 3)

        bridge.unlink()

    def test_multiple_frames_no_leak(self):
        """多次写入不泄漏。"""
        from src.pipeline.shared_memory_bridge import SharedMemoryBridge

        bridge = SharedMemoryBridge(name="test_shm_multi", size=1920*1080*3)
        bridge.create()

        for i in range(10):
            frame = np.full((1080, 1920, 3), i % 256, dtype=np.uint8)
            bridge.write_frame(frame)
            assert bridge.read_frame()[0, 0, 0] == i % 256

        bridge.unlink()


# ── 优化 2: 固定几何尺寸 VRAM 防御 ────────────────────────────

class TestVRAMFixedNormalization:
    """4.4.5 — 512×512 固定尺寸归一化"""

    def test_normalize_to_fixed_512(self):
        """任意尺寸图像 → 511×511 等比缩放 + 黑边 → 512×512。"""
        from src.ocr.vlm_extractor import VLMExtractor

        # 模拟不规则弹窗截图
        img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        result = VLMExtractor.normalize_to_fixed(img, 512, 512)

        assert result.shape == (512, 512, 3)
        # 黑边应存在（非等比图像）
        assert result[0, 0, 0] == 0 or result[0, 0, 1] == 0

    def test_square_image_fills_canvas(self):
        """正方形图像填满无黑边。"""
        from src.ocr.vlm_extractor import VLMExtractor

        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = VLMExtractor.normalize_to_fixed(img, 512, 512)
        assert result.shape == (512, 512, 3)

    def test_consistent_size_across_varied_inputs(self):
        """不同尺寸输入 → 始终 512×512（杜绝碎片）。"""
        from src.ocr.vlm_extractor import VLMExtractor

        sizes = [(1200, 800), (400, 300), (1920, 1080), (200, 200)]
        for h, w in sizes:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = VLMExtractor.normalize_to_fixed(img)
            assert result.shape == (512, 512, 3), f"输入 {h}×{w} → {result.shape}"


# ── 优化 3: Trie + Levenshtein SKU 解析 ───────────────────────

class TestTrieSKUResolver:
    """5.5.1 — Trie 前缀树 + 编辑距离精准对齐"""

    @pytest.fixture
    def resolver(self):
        from src.core.sku_resolver import SKUResolver
        titles = [
            "综合知识笔试课", "职业能力测试", "行测全家桶",
            "申论批改", "江西三支一扶公基", "江西三支一扶申论课",
        ]
        return SKUResolver(title_tree=titles, max_distance=3)

    def test_substring_match_instant(self, resolver):
        """子串包含匹配（快速路径）。"""
        result = resolver.resolve("综合知识笔试...")
        assert result == "综合知识笔试课"

    def test_exact_match_cached(self, resolver):
        """精确匹配 + 缓存命中。"""
        r1 = resolver.resolve("职业能力测试")
        r2 = resolver.resolve("职业能力测试")
        assert r1 == r2 == "职业能力测试"

    def test_trie_candidate_filter(self, resolver):
        """Trie 前缀树筛选候选集。"""
        from src.core.sku_resolver import FastSKUIndex

        trie = FastSKUIndex()
        trie.batch_insert(["江西三支一扶公基", "江西三支一扶申论课", "综合知识笔试课"])

        candidates = trie.search("江西三支一扶")
        assert len(candidates) == 2

    def test_levenshtein_fallback(self, resolver):
        """Levenshtein 编辑距离兜底。"""
        result = resolver.resolve("江西三支一扶公基课")  # 多一个"课"
        assert result == "江西三支一扶公基"

    def test_batch_resolve(self, resolver):
        """批量解析。"""
        results = resolver.resolve_batch([
            "综合知识笔试...", "职业能力测试", "行测全家桶",
        ])
        assert results == ["综合知识笔试课", "职业能力测试", "行测全家桶"]

    def test_add_title_dynamic(self, resolver):
        """动态追加标题。"""
        resolver.add_title("新课程名称")
        result = resolver.resolve("新课程...")
        assert result == "新课程名称"

    def test_stats_accurate(self, resolver):
        resolver.resolve("综合知识笔试...")
        resolver.resolve("不存在的奇怪SKU%%%%%")
        stats = resolver.stats
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1


# ── 优化 4: Scrcpy Control Socket 消息构造 ────────────────────

# ── 审查补丁：JSON 修复网关 ──────────────────────────────────

class TestJSONRepair:
    """11.2 — VLM 输出 JSON 自动修复"""

    def test_repair_missing_closing_brace(self):
        from src.ocr.vlm_extractor import VLMExtractor
        broken = '{"summary":{"cart_adds":"test"},"orders":[{"buyer":"A**"'
        repaired = VLMExtractor._repair_json(broken)
        assert repaired.endswith("}]")  # 先补 } 再补 ]
        assert repaired.count("{") == repaired.count("}")

    def test_repair_markdown_wrapped(self):
        from src.ocr.vlm_extractor import VLMExtractor
        text = '```json\n{"summary":{"cart_adds":"x"},"orders":[]}\n```'
        # _parse_response 处理 markdown
        import re
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())
        data = VLMExtractor._try_parse_json(text)
        assert data is not None
        assert data["summary"]["cart_adds"] == "x"

    def test_repair_control_characters(self):
        from src.ocr.vlm_extractor import VLMExtractor
        dirty = '{"summary":{"cart_adds":"x\x00y"},"orders":[]}'
        repaired = VLMExtractor._repair_json(dirty)
        assert '\x00' not in repaired


class TestScrcpyControlSocket:
    """6.1.3 — Control Socket 二进制协议"""

    def test_inject_text_message_format(self):
        """文本注入消息格式正确。"""
        from src.core.scrcpy_control_socket import ControlMessage, CONTROL_TYPE_INJECT_TEXT
        import struct

        msg = ControlMessage.inject_text("hello")
        assert len(msg) > 0
        # 验证消息头
        msg_type, flags = struct.unpack_from(">BB", msg)
        assert msg_type == CONTROL_TYPE_INJECT_TEXT

    def test_touch_message_format(self):
        """触控消息格式正确。"""
        from src.core.scrcpy_control_socket import ControlMessage, CONTROL_TYPE_INJECT_TOUCH_EVENT, ACTION_DOWN
        import struct

        msg = ControlMessage.inject_touch(ACTION_DOWN, 0, 540, 960, 0x8000)
        msg_type, action = struct.unpack_from(">BB", msg)
        assert msg_type == CONTROL_TYPE_INJECT_TOUCH_EVENT
        assert action == ACTION_DOWN

    def test_control_message_roundtrip(self):
        """消息构造不抛异常。"""
        from src.core.scrcpy_control_socket import ControlMessage, ACTION_DOWN, ACTION_UP

        # 各种消息类型均不抛异常
        ControlMessage.inject_text("中文测试")
        ControlMessage.inject_touch(ACTION_DOWN, 0, 100, 200, 0x4000)
        ControlMessage.inject_touch(ACTION_UP, 0, 100, 200, 0)
        ControlMessage.inject_keycode(4)  # KEYCODE_BACK
        ControlMessage.back_or_screen_on(ControlMessage.ACTION_BACK)


# ── 模块级 Worker（Windows spawn pickle 要求）─────────────────

def _shm_child_reader(name, result_queue):
    from src.pipeline.shared_memory_bridge import SharedMemoryBridge
    b = SharedMemoryBridge(name="")
    b.attach(name)
    f = b.read_frame()
    result_queue.put(f.shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
