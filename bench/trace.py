"""Event tracing.

Stages emit timestamped events; metrics are derived afterwards from the event
log. Keeping the two apart means any metric can be recomputed without re-running
a trial, and timing code never leaks into pipeline logic.

All timestamps come from `time.monotonic()`. Never use `time.time()` here: it is
subject to NTP adjustments and can move backwards.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Canonical event names. Systems should emit these so metrics.py can find them.
# Anything extra is allowed and ignored by the standard metrics.
FEEDER_START = ("feeder", "start")
FEEDER_ENDPOINT = ("feeder", "endpoint")   # t=0 for every latency metric
FEEDER_END = ("feeder", "end")

ASR_FIRST_PARTIAL = ("asr", "first_partial")
ASR_FINAL = ("asr", "final")

LLM_FIRST_TOKEN = ("llm", "first_token")
LLM_LAST_TOKEN = ("llm", "last_token")

TTS_FIRST_CHUNK = ("tts", "first_chunk")
TTS_LAST_CHUNK = ("tts", "last_chunk")

OUTPUT_FIRST_AUDIO = ("output", "first_audio")
OUTPUT_CHUNK = ("output", "chunk")
OUTPUT_END = ("output", "end")


@dataclass
class Event:
    stage: str
    event: str
    t: float                      # monotonic seconds
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """One trial: one clip through one system config."""

    clip_id: str
    config: str
    trial: int
    warmup: bool = False
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[Event] = field(default_factory=list)
    # Recorded so results from different hardware are never silently compared.
    env: dict[str, Any] = field(default_factory=dict)

    def mark(self, stage_event: tuple[str, str], **meta: Any) -> float:
        stage, event = stage_event
        t = time.monotonic()
        self.events.append(Event(stage=stage, event=event, t=t, meta=meta))
        return t

    def mark_at(self, stage_event: tuple[str, str], t: float, **meta: Any) -> float:
        """Records an event at an explicit monotonic time.

        Used for instants that are computed rather than observed, such as the
        scheduled endpoint, so the timestamp does not drift with when the loop
        happened to notice it.
        """
        stage, event = stage_event
        self.events.append(Event(stage=stage, event=event, t=t, meta=meta))
        return t

    def first(self, stage_event: tuple[str, str]) -> float | None:
        stage, event = stage_event
        for e in self.events:
            if e.stage == stage and e.event == event:
                return e.t
        return None

    def all_of(self, stage_event: tuple[str, str]) -> list[Event]:
        stage, event = stage_event
        return [e for e in self.events if e.stage == stage and e.event == event]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


class TraceWriter:
    """Appends traces to a JSONL file, one trace per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: Trace) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def load_traces(path: str | Path) -> list[Trace]:
    traces: list[Trace] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            events = [Event(**e) for e in d.pop("events", [])]
            traces.append(Trace(events=events, **d))
    return traces
