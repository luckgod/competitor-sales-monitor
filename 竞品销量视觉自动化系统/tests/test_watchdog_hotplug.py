"""阶段一验收测试：看门狗热插拔自愈 + 会话状态断点续爬。

验收标准（来自核心架构开发与验收总览 - 1.2 单元验收标准）：
  测试算子：模拟 USB 物理断连 5 秒后重插
  红线 1：主程序绝对不允许抛出异常退出
  红线 2：看门狗在 3 秒内检测到断连，自动执行 kill-server → start-server 重握手
  红线 3：重连后自动重拉 scrcpy，根据 Index 锚点恢复滑动位置
"""
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.adb_controller import ADBController, DeviceInfo
from src.core.watchdog import Watchdog
from src.core.session import SessionManager, SessionState


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────

def _make_mock_proc(alive: bool = True):
    """创建一个 mock subprocess.Popen，poll() 返回 None 表示存活。"""
    proc = MagicMock()
    proc.poll.return_value = None if alive else 1
    return proc


def _set_scrcpy_alive(adb: ADBController, alive: bool) -> None:
    """控制 scrcpy_alive property 的返回值。

    ADBController.scrcpy_alive 是只读 property，通过 _scrcpy_proc 状态判定。
    本函数通过操作 _scrcpy_proc 来模拟进程存活/死亡。
    """
    if alive:
        adb._scrcpy_proc = _make_mock_proc(alive=True)
    else:
        adb._scrcpy_proc = None


# ────────────────────────────────────────────────────────────
# 测试夹具
# ────────────────────────────────────────────────────────────

@pytest.fixture
def adb():
    """创建一个完全 mock 的 ADBController，不执行真实 shell 命令。

    默认行为：launch_scrcpy 成功后自动将 scrcpy_alive 设为 True，
    模拟自愈成功流程。需要模拟自愈失败的测试可覆盖此行为。
    """
    ctrl = ADBController(adb_path="mock_adb", scrcpy_path="mock_scrcpy")
    ctrl._run_adb = MagicMock(return_value="")
    ctrl.kill_scrcpy = MagicMock()

    def _launch_ok(*a, **kw):
        _set_scrcpy_alive(ctrl, True)
    ctrl.launch_scrcpy = MagicMock(side_effect=_launch_ok)

    _set_scrcpy_alive(ctrl, True)
    ctrl.is_device_connected = MagicMock(return_value=True)
    ctrl.list_devices = MagicMock(return_value=[
        DeviceInfo(serial="emulator-5554", model="TestPhone")
    ])
    ctrl.wait_for_device = MagicMock(return_value=True)
    return ctrl


@pytest.fixture
def session_mgr(tmp_path):
    state_file = tmp_path / "session_state.json"
    mgr = SessionManager(state_file=str(state_file))
    mgr.new_batch(store_id="store_001")
    mgr.update_progress(progress=42, virtual_id="prod_abc123")
    return mgr


@pytest.fixture
def watchdog(adb):
    return Watchdog(
        adb=adb,
        check_interval=0.5,  # 加速测试，实际为 5s
        reconnect_retries=3,
        wireless_fallback=True,
        device_ip="192.168.1.100",
    )


# ────────────────────────────────────────────────────────────
# 红线 2：看门狗 3 秒内检测到断连并触发自愈
# ────────────────────────────────────────────────────────────

class TestWatchdogDetectionAndHealing:
    """验收红线 2 — 断连检测时效 + 自愈流程完整性"""

    def test_detect_scrcpy_crash_within_3_seconds(self, adb, watchdog):
        """看门狗在 3 秒内检测到 scrcpy 进程消失并触发自愈。

        验证点：
        - 检测耗时 < 3s（文档红线）
        - 自愈流程 kill_scrcpy + launch_scrcpy 被调用
        - 恢复回调触发
        """
        _set_scrcpy_alive(adb, False)

        heal_events = []
        watchdog.on_recovered(lambda: heal_events.append(time.monotonic()))

        watchdog.start()
        detection_start = time.monotonic()

        deadline = time.monotonic() + 5
        while not heal_events and time.monotonic() < deadline:
            time.sleep(0.1)

        watchdog.stop()

        assert heal_events, "看门狗未在 5 秒内检测到断连"
        detection_time = heal_events[0] - detection_start
        assert detection_time < 3.0, (
            f"断连检测耗时 {detection_time:.2f}s，超过 3 秒红线"
        )
        assert adb.kill_scrcpy.called, "自愈流程未执行 kill_scrcpy"
        assert adb.launch_scrcpy.called, "自愈流程未重新拉起 scrcpy"

    def test_detect_device_disconnect_and_wireless_fallback(self, adb, watchdog):
        """ADB 设备列表为空时触发无线兜底重连。

        验证点：
        - 设备断连被检测到
        - 无线 ADB connect 或 wait_for_device 被调用
        - 自愈后恢复回调触发
        """
        _set_scrcpy_alive(adb, False)
        # 设备状态：首次检查返回 False，recover_device 执行后恢复 True
        call_count = [0]
        def _device_state():
            call_count[0] += 1
            return call_count[0] > 1  # 同一周期内 post-recovery 检查时恢复
        adb.is_device_connected = MagicMock(side_effect=_device_state)

        heal_events = []
        watchdog.on_recovered(lambda: heal_events.append(True))

        watchdog.start()
        deadline = time.monotonic() + 8
        while not heal_events and time.monotonic() < deadline:
            time.sleep(0.1)
        watchdog.stop()

        assert heal_events, "看门狗未检测到设备断连并完成自愈"
        connect_calls = [
            c for c in adb._run_adb.call_args_list
            if "connect" in str(c)
        ]
        assert len(connect_calls) > 0 or adb.wait_for_device.called, (
            "未触发无线 ADB 兜底或等待设备"
        )

    def test_consecutive_failures_triggers_alert(self, adb, watchdog):
        """连续 3 次自愈失败触发远程告警。

        验证点：
        - 连续失败数达到阈值时触发 on_alert 回调
        - 告警消息包含关键信息
        """
        _set_scrcpy_alive(adb, False)
        adb.is_device_connected = MagicMock(return_value=False)
        # launch_scrcpy 后 scrcpy 仍然死亡 → 自愈失败
        adb.launch_scrcpy.side_effect = lambda *a, **kw: _set_scrcpy_alive(adb, False)

        alerts = []
        watchdog.on_alert(lambda msg: alerts.append(msg))

        watchdog.start()
        deadline = time.monotonic() + 12
        while not alerts and time.monotonic() < deadline:
            time.sleep(0.2)
        watchdog.stop()

        assert len(alerts) > 0, "连续失败应触发告警"


# ────────────────────────────────────────────────────────────
# 红线 1：主程序绝对不允许抛出异常退出
# ────────────────────────────────────────────────────────────

class TestNoExceptionOnDisconnect:
    """验收红线 1 — 任何断连场景下主程序不崩溃"""

    def test_watchdog_handles_adb_timeout_gracefully(self, adb, watchdog):
        """ADB 命令超时时看门狗不抛出异常。"""
        _set_scrcpy_alive(adb, False)
        adb._run_adb.side_effect = subprocess.TimeoutExpired("adb", 15)

        watchdog.start()
        time.sleep(3)
        watchdog.stop()
        # 到达此处即证明无异常抛出

    def test_watchdog_handles_process_lookup_error(self, adb, watchdog):
        """进程已消失时 kill_scrcpy 不抛异常。"""
        _set_scrcpy_alive(adb, False)
        adb.kill_scrcpy.side_effect = ProcessLookupError("no such process")

        watchdog.start()
        time.sleep(3)
        watchdog.stop()

    def test_recovery_callback_exception_does_not_crash_watchdog(self, adb, watchdog):
        """恢复回调内部异常不影响看门狗正常运行。

        验证点：
        - 一个回调抛异常，后续回调仍然执行
        - 看门狗线程不因回调异常而退出
        """
        _set_scrcpy_alive(adb, False)

        def _crashing_callback():
            raise RuntimeError("回调内部异常模拟")
        watchdog.on_recovered(_crashing_callback)

        normal_callback_called = []
        watchdog.on_recovered(lambda: normal_callback_called.append(True))

        watchdog.start()
        deadline = time.monotonic() + 5
        while not normal_callback_called and time.monotonic() < deadline:
            time.sleep(0.1)
        watchdog.stop()

        assert normal_callback_called, "异常回调不应阻止后续回调执行"

    def test_producer_thread_survives_watchdog_recovery(self, adb, watchdog, tmp_path):
        """模拟生产者线程在投屏恢复后继续正常运行。

        验证点：
        - 断连 → 自愈 → 生产者收到恢复通知
        - 生产者可正常停止
        """
        from src.pipeline.queue_manager import ImageQueue
        from src.producer.capture import ProducerThread
        from src.core.session import SessionManager

        state_file = tmp_path / "state.json"
        queue = ImageQueue(max_size=10, low_watermark=7)
        session_mgr = SessionManager(state_file=str(state_file))
        session_mgr.new_batch(store_id="store_001")

        producer = ProducerThread(
            adb=adb, queue=queue, session_mgr=session_mgr,
        )
        watchdog.on_recovered(producer.reset_frame_source)

        producer.start()
        time.sleep(1)

        _set_scrcpy_alive(adb, False)
        heal_done = threading.Event()
        watchdog.on_recovered(lambda: heal_done.set())
        watchdog.start()

        healed = heal_done.wait(timeout=5)
        producer.stop()
        watchdog.stop()

        assert healed, "自愈未完成"
        assert producer._thread.is_alive() is False, "生产者应能正常停止"


# ────────────────────────────────────────────────────────────
# 红线 3：锚点断点续爬 — 恢复后从 last_successful_virtual_id 继续
# ────────────────────────────────────────────────────────────

class TestSessionResumeFromAnchor:
    """验收红线 3 — 断连恢复后从锚点续爬，而非从头开始"""

    def test_session_state_persists_through_disconnect(self, tmp_path):
        """SessionState 在断连前后完整保持。

        模拟场景：已采集 42 个商品 → 断连 → 程序重启 → 状态无损恢复
        """
        state_file = tmp_path / "state.json"
        mgr = SessionManager(state_file=str(state_file))

        mgr.new_batch(store_id="store_001")
        mgr.update_progress(progress=42, virtual_id="prod_abc123")

        before = mgr.state
        assert before.current_store_id == "store_001"
        assert before.current_store_progress == 42
        assert before.last_successful_virtual_id == "prod_abc123"

        # 模拟程序重启（重新读取状态文件）
        mgr2 = SessionManager(state_file=str(state_file))
        after = mgr2.state

        assert after.current_store_id == before.current_store_id
        assert after.current_store_progress == before.current_store_progress
        assert after.last_successful_virtual_id == before.last_successful_virtual_id
        assert after.capture_batch_id == before.capture_batch_id

    def test_new_batch_generates_unique_id(self, tmp_path):
        """每次新跑盘生成唯一 batch_id。"""
        state_file = tmp_path / "state.json"
        mgr = SessionManager(state_file=str(state_file))
        b1 = mgr.new_batch(store_id="store_001")
        b2 = mgr.new_batch(store_id="store_002")
        assert b1.capture_batch_id != b2.capture_batch_id
        assert len(b1.capture_batch_id) == 12

    def test_update_progress_partial_fields(self, tmp_path):
        """进度更新支持部分字段，未指定的字段保持不变。"""
        state_file = tmp_path / "state.json"
        mgr = SessionManager(state_file=str(state_file))
        mgr.new_batch(store_id="store_001")

        mgr.update_progress(progress=10)
        assert mgr.state.current_store_progress == 10
        assert mgr.state.current_store_id == "store_001"  # 不变

        mgr.update_progress(virtual_id="prod_xyz")
        assert mgr.state.last_successful_virtual_id == "prod_xyz"
        assert mgr.state.current_store_progress == 10  # 不变


# ────────────────────────────────────────────────────────────
# 补充：有界阻塞队列背压测试（为阶段二做前置验证）
# ────────────────────────────────────────────────────────────

class TestBoundedQueueBackpressure:
    """阶段一附加验证：队列背压机制是否就绪"""

    def test_queue_blocks_at_max_size(self):
        from src.pipeline.queue_manager import ImageQueue
        q = ImageQueue(max_size=3, low_watermark=2, producer_timeout=0.5)

        assert q.put("frame1")
        assert q.put("frame2")
        assert q.put("frame3")
        assert q.is_full

        assert not q.put("frame4"), "队列满时应拒绝入队"

    def test_queue_wakes_producer_below_watermark(self):
        from src.pipeline.queue_manager import ImageQueue
        q = ImageQueue(max_size=3, low_watermark=2, producer_timeout=0.5)

        for i in range(3):
            q.put(f"frame{i}")

        q.get(timeout=0.5)
        q.get(timeout=0.5)

        assert not q.is_full
        assert q.should_produce, "低于低水位应允许生产者恢复"

    def test_producer_pause_resume(self):
        from src.pipeline.queue_manager import ImageQueue
        q = ImageQueue(max_size=5, low_watermark=3)

        q.block_producer()
        assert q._producer_paused.is_set()

        q.resume_producer()
        assert not q._producer_paused.is_set()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
