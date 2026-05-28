"""采集会话状态持久化 — 支持断线后无状态恢复，从锚点续爬而非从头开始。"""
import json
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SessionState:
    current_store_id: str = ""
    current_store_progress: int = 0
    last_successful_virtual_id: str = ""
    capture_batch_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionManager:
    """管理采集会话的持久化与恢复。"""

    def __init__(self, state_file: str = "state/session_state.json"):
        self._state_file = Path(state_file)
        self._state: SessionState | None = None

    @property
    def state(self) -> SessionState:
        if self._state is None:
            self._state = self._load_or_create()
        return self._state

    def new_batch(self, store_id: str = "") -> SessionState:
        self._state = SessionState(
            current_store_id=store_id,
            current_store_progress=0,
            last_successful_virtual_id="",
            capture_batch_id=uuid.uuid4().hex[:12],
        )
        self._persist()
        return self._state

    def update_progress(self, store_id: str | None = None,
                        progress: int | None = None,
                        virtual_id: str | None = None) -> None:
        s = self.state
        if store_id is not None:
            s.current_store_id = store_id
        if progress is not None:
            s.current_store_progress = progress
        if virtual_id is not None:
            s.last_successful_virtual_id = virtual_id
        self._persist()

    def _load_or_create(self) -> SessionState:
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                return SessionState.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass
        return SessionState()

    def _persist(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(
            json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
