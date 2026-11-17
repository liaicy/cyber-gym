from __future__ import annotations

import json
import threading
from pathlib import Path

from .schemas import ItemRecord


class RunStore:
    """Append-only JSONL persistence with resume support."""

    def __init__(self, out_dir: str | Path):
        # Items are generated concurrently, so two threads can finish at the
        # same moment. Without this lock their JSON lines interleave and the
        # file stops being parseable -- silently, and only for the rows written
        # during the collision.
        self._lock = threading.Lock()
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.items_path = self.dir / "items.jsonl"
        self.state_path = self.dir / "state.json"

    def append(self, record: ItemRecord) -> None:
        # Serialise outside the lock, write inside it: two concurrent items
        # finishing together would otherwise interleave their JSON lines and
        # corrupt the file for exactly those rows, silently.
        line = json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
        with self._lock, self.items_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_all(self) -> list[ItemRecord]:
        if not self.items_path.exists():
            return []
        records = []
        with self.items_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(ItemRecord.model_validate(json.loads(line)))
        return records

    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def save_state(self, state: dict) -> None:
        with self._lock:
            self._write_state(state)

    def _write_state(self, state: dict) -> None:
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_records(path: str | Path) -> list[ItemRecord]:
    """Load records from an items.jsonl file or a run directory."""
    p = Path(path)
    if p.is_dir():
        p = p / "items.jsonl"
    records = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ItemRecord.model_validate(json.loads(line)))
    return records
