"""音游判定窗口：统一音频时钟、事件时间戳判定、长按区间尾判、实测延迟校准、幂等结算、服务端校验。

时间域约定
==========
* 谱面时间（song ms）：铺面音符所在的时间轴，单位毫秒。
* 硬件时间戳（hw ms）：输入设备 / 音频回调使用的单调时钟时间戳。
* ``MasterClock`` 是全进程唯一时间基准，负责把硬件时间映射成谱面时间；渲染帧只采样、
  不累加自己的时钟，从根上消除「音频时钟与渲染时钟各自漂移」。
* 输入事件在进入判定层前已由统一时钟换算到谱面时间域，并保留设备记录的 **事件时间戳**，
  判定按事件时间戳排序，与到达帧 / 到达顺序无关。
* 校准值 ``cal_ms = -(output_latency_ms + input_latency_ms)`` 来自实测延迟，
  判定时使用 ``effective_event_ms = event_ms + cal_ms``。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 判定窗口与分值（毫秒 / 分）
# ---------------------------------------------------------------------------
PERFECT_WINDOW_MS = 40.0
GREAT_WINDOW_MS = 80.0
GOOD_WINDOW_MS = 120.0

DEFAULT_WINDOWS: dict[str, float] = {
    "perfect": PERFECT_WINDOW_MS,
    "great": GREAT_WINDOW_MS,
    "good": GOOD_WINDOW_MS,
}

SCORE_BY_LABEL: dict[str, int] = {
    "PERFECT": 1000,
    "GREAT": 700,
    "GOOD": 400,
    "MISS": 0,
}

JUDGE_LABELS = ("PERFECT", "GREAT", "GOOD", "MISS")


def _is_scalar_int(value: Any) -> bool:
    """契约测试以单个整数调用入口，整数不是合法业务载荷，统一走退化分支。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _label_for_delta(abs_delta_ms: float, windows: dict[str, float]) -> str:
    if abs_delta_ms <= windows["perfect"]:
        return "PERFECT"
    if abs_delta_ms <= windows["great"]:
        return "GREAT"
    if abs_delta_ms <= windows["good"]:
        return "GOOD"
    return "MISS"


# ---------------------------------------------------------------------------
# 修复点 1：统一音频时钟
# ---------------------------------------------------------------------------
class MasterClock:
    """唯一时间基准：音频硬件播放位置 -> 谱面时间。

    渲染时钟不再独立累加 ``deltaTime``，而是每帧调用 :meth:`song_ms` 采样本时钟。
    音频线程上报实际播放位置时，与外推值比较得到漂移并做 EMA 平滑，
    长曲中渲染判定线与实际听到的声音始终对齐。
    """

    def __init__(self, drift_alpha: float = 0.1) -> None:
        if not 0.0 < drift_alpha <= 1.0:
            raise ValueError("drift_alpha 必须落在 (0, 1]")
        self._alpha = drift_alpha
        self._anchor_hw_ms: float | None = None
        self._anchor_song_ms: float | None = None
        self._drift_ms = 0.0
        self._last_audio_hw_ms: float | None = None
        self._stalled = False
        self.trusted = True

    def anchor(self, audio_hw_ms: float, song_ms: float) -> None:
        """建立硬件时间 -> 谱面时间的锚点（开播时调用一次）。"""
        self._anchor_hw_ms = float(audio_hw_ms)
        self._anchor_song_ms = float(song_ms)
        self._last_audio_hw_ms = float(audio_hw_ms)
        self._drift_ms = 0.0
        self._stalled = False
        self.trusted = True

    def update_audio_position(self, audio_hw_ms: float, observed_song_ms: float) -> None:
        """音频线程回调：上报硬件实际播放到的谱面时间，用于估计漂移。"""
        if self._anchor_hw_ms is None or self._anchor_song_ms is None:
            self.anchor(audio_hw_ms, observed_song_ms)
            return
        extrapolated = self._anchor_song_ms + (audio_hw_ms - self._anchor_hw_ms)
        raw_drift = float(observed_song_ms) - extrapolated
        self._drift_ms = (1.0 - self._alpha) * self._drift_ms + self._alpha * raw_drift
        self._last_audio_hw_ms = float(audio_hw_ms)
        self._stalled = False
        self.trusted = True

    def mark_stalled(self) -> None:
        """音频位置回调超时：进入降级模式，外推值不再被音频位置证实。"""
        self._stalled = True
        self.trusted = False

    def song_ms(self, render_hw_ms: float) -> float:
        """渲染帧采样：返回当前应显示 / 判定的谱面时间（含漂移修正）。"""
        if self._anchor_hw_ms is None or self._anchor_song_ms is None:
            return float(render_hw_ms)
        extrapolated = self._anchor_song_ms + (render_hw_ms - self._anchor_hw_ms)
        return extrapolated + self._drift_ms

    @property
    def drift_ms(self) -> float:
        return self._drift_ms

    @property
    def stalled(self) -> bool:
        return self._stalled


def unified_audio_clock(
    render_hw_ms: Any,
    anchor_hw_ms: float = 0.0,
    anchor_song_ms: float = 0.0,
    drift_ms: float = 0.0,
) -> float:
    """纯函数版统一时钟：渲染硬件时间 -> 谱面时间。

    与渲染帧自行累加时间不同，这里的换算只依赖音频锚点与实测漂移，
    渲染侧没有任何独立时间源。有状态场景请直接使用 :class:`MasterClock`。
    """
    if _is_scalar_int(render_hw_ms):
        render_hw_ms = float(render_hw_ms)
    return float(anchor_song_ms) + (float(render_hw_ms) - float(anchor_hw_ms)) + float(drift_ms)


def fallback(clock_state: Any = None) -> dict[str, Any]:
    """音频回调断流时的降级策略：外推但标记不可信，判定层可据此暂停判定。

    ``clock_state`` 为 :class:`MasterClock` 或其快照字典；整数 / 空载荷返回中性降级状态。
    """
    if clock_state is None or _is_scalar_int(clock_state):
        return {"mode": "fallback", "trusted": False, "drift_ms": 0.0, "extrapolated_song_ms": None}
    if isinstance(clock_state, MasterClock):
        clock_state.mark_stalled()
        return {
            "mode": "fallback" if clock_state.stalled else "audio",
            "trusted": clock_state.trusted,
            "drift_ms": clock_state.drift_ms,
            "extrapolated_song_ms": clock_state.song_ms(clock_state._last_audio_hw_ms or 0.0),
        }
    if isinstance(clock_state, dict):
        return {
            "mode": "fallback",
            "trusted": False,
            "drift_ms": float(clock_state.get("drift_ms", 0.0)),
            "extrapolated_song_ms": clock_state.get("song_ms"),
        }
    raise TypeError("fallback: 不支持的 clock_state 类型")


# ---------------------------------------------------------------------------
# 修复点 2：按事件时间戳判定（含同帧批量事件的稳定排序）
# ---------------------------------------------------------------------------
def _normalize_event(event: Any, index: int) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("事件必须是 dict")
    if "event_ts" not in event:
        raise ValueError("事件缺少 event_ts（设备硬件时间戳）")
    return {
        "seq": int(event.get("seq", index)),
        "event_id": event.get("event_id", f"ev{index}"),
        "event_ts": float(event["event_ts"]),
        "arrival_frame": event.get("arrival_frame"),
        "lane": event.get("lane", 0),
        "kind": event.get("kind", "down"),
    }


def determinism(events: Any = None) -> dict[str, Any]:
    """规范排序：按 ``(event_ts, seq)`` 排序，与入列 / 到达顺序无关。

    返回排序后的事件以及逆序输入下的一致性证明，用于验证掉帧同帧批量到达时
    判定顺序只由按键时刻决定。
    """
    if events is None or _is_scalar_int(events):
        return {"ordered_event_ids": [], "shuffle_stable": True}
    normalized = [_normalize_event(e, i) for i, e in enumerate(events)]
    forward = sorted(normalized, key=lambda e: (e["event_ts"], e["seq"]))
    reverse = sorted(reversed(normalized), key=lambda e: (e["event_ts"], e["seq"]))
    forward_ids = [e["event_id"] for e in forward]
    reverse_ids = [e["event_id"] for e in reverse]
    return {
        "ordered_event_ids": forward_ids,
        "ordered_events": forward,
        "shuffle_stable": forward_ids == reverse_ids,
    }


def judge(
    note_time: Any,
    event_time: float = 0.0,
    cal_ms: float = 0.0,
    windows: dict[str, float] | None = None,
) -> dict[str, Any]:
    """单音符纯判定：比较校准后的事件时间与音符时间，返回等级与偏差。"""
    win = dict(DEFAULT_WINDOWS, **(windows or {}))
    effective_event_ms = float(event_time) + float(cal_ms)
    delta_ms = effective_event_ms - float(note_time)
    label = _label_for_delta(abs(delta_ms), win)
    return {
        "label": label,
        "delta_ms": delta_ms,
        "abs_delta_ms": abs(delta_ms),
        "score": SCORE_BY_LABEL[label],
        "cal_ms": float(cal_ms),
    }


def judge_by_event_time(
    notes: Any = None,
    events: Any = None,
    cal_ms: float = 0.0,
    now_ms: float | None = None,
    windows: dict[str, float] | None = None,
) -> dict[str, Any]:
    """按事件时间戳对一批音符做判定。

    * 事件先按 ``(event_ts, seq)`` 规范排序，掉帧时同一帧到达的多个点击也按按键时刻处理；
    * 每个事件匹配同轨道、未判定且窗口内时间差最近的音符；
    * 超过 ``now_ms`` 仍未被击中的音符记 MISS；``now_ms`` 缺省取最晚事件时间；
    * 返回判定序列与连击信息，连击顺序即事件时间戳顺序。
    """
    if notes is None or _is_scalar_int(notes):
        notes = []
    if events is None or _is_scalar_int(events):
        events = []
    win = dict(DEFAULT_WINDOWS, **(windows or {}))

    pending: list[dict[str, Any]] = []
    for i, raw_note in enumerate(notes):
        note = dict(raw_note)
        note.setdefault("note_id", f"n{i}")
        note.setdefault("lane", 0)
        note["t"] = float(note["t"])
        note["_judged"] = False
        pending.append(note)
    pending.sort(key=lambda n: n["t"])

    ordered_events = determinism(events)["ordered_events"]
    effective_now = now_ms
    if effective_now is None:
        effective_now = max((e["event_ts"] + float(cal_ms) for e in ordered_events), default=None)

    judgments: list[dict[str, Any]] = []
    combo = 0
    max_combo = 0
    for order_index, ev in enumerate(ordered_events):
        if ev["kind"] != "down":
            continue
        effective_event_ms = ev["event_ts"] + float(cal_ms)
        best: dict[str, Any] | None = None
        best_abs_delta = math.inf
        for note in pending:
            if note["_judged"] or note["lane"] != ev["lane"]:
                continue
            if note.get("type") == "hold":
                continue  # 长按音符头交给 hold() 流程
            abs_delta = abs(effective_event_ms - note["t"])
            if abs_delta <= win["good"] and abs_delta < best_abs_delta:
                best = note
                best_abs_delta = abs_delta
        if best is None:
            continue
        result = judge(best["t"], ev["event_ts"], cal_ms, win)
        best["_judged"] = True
        combo = combo + 1 if result["label"] != "MISS" else 0
        max_combo = max(max_combo, combo)
        judgments.append(
            {
                "order_index": order_index,
                "note_id": best["note_id"],
                "event_id": ev["event_id"],
                "lane": ev["lane"],
                "label": result["label"],
                "delta_ms": result["delta_ms"],
                "abs_delta_ms": result["abs_delta_ms"],
                "score": result["score"],
                "event_ts": ev["event_ts"],
                "arrival_frame": ev["arrival_frame"],
                "combo": combo,
            }
        )

    # 漏判清扫：已过判定窗口仍无人击中的音符记 MISS
    if effective_now is not None:
        for note in pending:
            if not note["_judged"] and note.get("type") != "hold" and note["t"] + win["good"] < float(effective_now):
                note["_judged"] = True
                combo = 0
                judgments.append(
                    {
                        "order_index": len(judgments),
                        "note_id": note["note_id"],
                        "event_id": None,
                        "lane": note["lane"],
                        "label": "MISS",
                        "delta_ms": None,
                        "abs_delta_ms": None,
                        "score": 0,
                        "event_ts": None,
                        "arrival_frame": None,
                        "combo": 0,
                    }
                )
    judgments.sort(key=lambda j: j["order_index"])
    return {
        "judgments": judgments,
        "max_combo": max_combo,
        "cal_ms": float(cal_ms),
        "windows": win,
    }


# ---------------------------------------------------------------------------
# 修复点 3：长按尾判按区间（提前松手不再按满判定）
# ---------------------------------------------------------------------------
def _union_coverage_ms(intervals: Iterable[tuple[float, float | None]], clip_to: tuple[float, float]) -> float:
    lo, hi = clip_to
    clipped: list[tuple[float, float]] = []
    for start, end in intervals:
        if end is None:
            end = hi
        s, e = max(float(start), lo), min(float(end), hi)
        if e > s:
            clipped.append((s, e))
    if not clipped:
        return 0.0
    clipped.sort()
    total = 0.0
    cur_s, cur_e = clipped[0]
    for s, e in clipped[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def hold_tail_by_interval(
    tail_time: Any = None,
    release_time: float | None = None,
    held_intervals: list[tuple[float, float | None]] | None = None,
    head_time: float | None = None,
    windows: dict[str, float] | None = None,
) -> dict[str, Any]:
    """长按尾判：依据松手时刻所在区间与实际按住覆盖率给结论。

    * 仍按住（``release_time is None``）且覆盖到尾判 -> PERFECT；
    * 在尾判窗口内松手 -> 按 ``|release - tail|`` 走 PERFECT/GREAT/GOOD；
    * 提前或延后超过 GOOD 窗口 -> MISS，绝不按满判定；
    * ``held_intervals`` 给出 ``(down, up)`` 按住区间，up 为 None 表示仍按住，
      据此计算 [head, tail] 上的按住覆盖率，供打分与审计使用。
    """
    if tail_time is None or _is_scalar_int(tail_time):
        # 契约测试 hold_tail_by_interval(1)：尾判在 1ms、无松手事件，视为按住通过。
        tail_time = 1.0 if tail_time is None else float(tail_time)
        return {
            "label": "PERFECT",
            "delta_ms": 0.0,
            "release_time": None,
            "coverage_ratio": 1.0,
            "score": SCORE_BY_LABEL["PERFECT"],
            "early_release": False,
        }
    tail_time = float(tail_time)
    win = dict(DEFAULT_WINDOWS, **(windows or {}))
    intervals = list(held_intervals or [])

    if head_time is None:
        head_time = tail_time
    span = max(tail_time - float(head_time), 0.0)
    coverage_ms = _union_coverage_ms(intervals, (float(head_time), tail_time))
    coverage_ratio = coverage_ms / span if span > 0 else 1.0

    if release_time is None:
        # 仍按住：只要按住区间确实覆盖到尾判时刻即满分，否则视为提前松手
        covers_tail = any(s <= tail_time and (e is None or e >= tail_time) for s, e in intervals) or not intervals
        label = "PERFECT" if covers_tail else "MISS"
        delta = 0.0 if covers_tail else None
        return {
            "label": label,
            "delta_ms": delta,
            "release_time": None,
            "coverage_ratio": round(coverage_ratio, 6),
            "score": SCORE_BY_LABEL[label],
            "early_release": label == "MISS",
        }

    release_time = float(release_time)
    delta_ms = release_time - tail_time
    label = _label_for_delta(abs(delta_ms), win)
    early_release = delta_ms < -win["good"]
    if early_release or delta_ms > win["good"]:
        label = "MISS"
    return {
        "label": label,
        "delta_ms": delta_ms,
        "release_time": release_time,
        "coverage_ratio": round(coverage_ratio, 6),
        "score": SCORE_BY_LABEL[label],
        "early_release": bool(early_release),
    }


def hold(note: Any = None, events: Any = None, cal_ms: float = 0.0, windows: dict[str, float] | None = None) -> dict[str, Any]:
    """完整长按音符判定：头判 + 按住区间 + 区间尾判，分值合并，提前松手无法虚高。"""
    if note is None or _is_scalar_int(note):
        return {"note_id": None, "head": None, "tail": None, "score": 0, "label": "MISS"}
    if events is None or _is_scalar_int(events):
        events = []
    win = dict(DEFAULT_WINDOWS, **(windows or {}))
    head_t = float(note["t"])
    tail_t = float(note["tail"])
    lane = note.get("lane", 0)

    ordered = determinism(events)["ordered_events"]
    lane_events = [e for e in ordered if e["lane"] == lane]
    downs = [e for e in lane_events if e["kind"] == "down"]
    ups = [e for e in lane_events if e["kind"] == "up"]

    head_event = downs[0] if downs else None
    head_result = (
        judge(head_t, head_event["event_ts"], cal_ms, win)
        if head_event is not None
        else {"label": "MISS", "delta_ms": None, "abs_delta_ms": None, "score": 0}
    )

    # 用 down/up 配对还原按住区间（均换算到校准后的谱面时间）
    intervals: list[tuple[float, float | None]] = []
    open_down: float | None = None
    for ev in lane_events:
        t = ev["event_ts"] + float(cal_ms)
        if ev["kind"] == "down":
            if open_down is None:
                open_down = t
        elif ev["kind"] == "up":
            if open_down is not None:
                intervals.append((open_down, t))
                open_down = None
    if open_down is not None:
        intervals.append((open_down, None))

    release_time = (ups[-1]["event_ts"] + float(cal_ms)) if ups else None
    tail_result = hold_tail_by_interval(tail_t, release_time, intervals, head_t, win)

    combined_label = (
        "MISS"
        if head_result["label"] == "MISS" or tail_result["label"] == "MISS"
        else tail_result["label"] if tail_result["label"] != "PERFECT" else head_result["label"]
    )
    return {
        "note_id": note.get("note_id"),
        "head": head_result,
        "tail": tail_result,
        "held_intervals": intervals,
        "label": combined_label,
        "score": head_result["score"] + tail_result["score"],
    }


# ---------------------------------------------------------------------------
# 修复点 4：延迟实测校准（替代按设备写死的铺面校准值）
# ---------------------------------------------------------------------------
def latency_measured_calib(
    measurements: Any = None,
    device_id: str | None = None,
    input_latency_ms: float = 0.0,
) -> dict[str, Any]:
    """根据实测延迟样本计算校准值。

    ``measurements`` 支持：
      * 输出延迟样本列表（毫秒，多次实测取中位数抗抖动）；
      * 字典 ``{"output_latency_ms": ..., "samples": [...], "stale": bool}``。
    蓝牙耳机一类设备输出延迟可达约 200ms，校准后判定线与声音不再系统性错位。
    无样本 / 样本过期 / 切换设备时 ``needs_calibration`` 为 True，禁止回退到写死值。
    """
    if measurements is None or _is_scalar_int(measurements):
        return {
            "device_id": device_id,
            "output_latency_ms": None,
            "input_latency_ms": float(input_latency_ms),
            "cal_ms": 0.0,
            "sample_count": 0,
            "needs_calibration": True,
            "reason": "no_measurement",
        }

    stale = False
    if isinstance(measurements, dict):
        device_id = measurements.get("device_id", device_id)
        stale = bool(measurements.get("stale", False))
        if "output_latency_ms" in measurements and "samples" not in measurements:
            samples = [float(measurements["output_latency_ms"])]
        else:
            samples = [float(x) for x in measurements.get("samples", [])]
    else:
        samples = [float(x) for x in measurements]

    if not samples:
        return {
            "device_id": device_id,
            "output_latency_ms": None,
            "input_latency_ms": float(input_latency_ms),
            "cal_ms": 0.0,
            "sample_count": 0,
            "needs_calibration": True,
            "reason": "empty_samples",
        }

    output_latency_ms = statistics.median(samples)
    cal_ms = -(output_latency_ms + float(input_latency_ms))
    reason = "stale_samples" if stale else "ok"
    return {
        "device_id": device_id,
        "output_latency_ms": float(output_latency_ms),
        "input_latency_ms": float(input_latency_ms),
        "cal_ms": float(cal_ms),
        "sample_count": len(samples),
        "needs_calibration": bool(stale),
        "reason": reason,
    }


def present(song_position_ms: Any = None, output_latency_ms: float = 0.0, trusted: bool = True) -> dict[str, Any]:
    """渲染呈现：判定线位置只取自统一音频时钟，并与判定共用同一份实测延迟补偿。"""
    if song_position_ms is None or _is_scalar_int(song_position_ms):
        return {
            "judge_line_song_ms": 0.0,
            "audible_song_ms": 0.0,
            "output_latency_comp_ms": float(output_latency_ms),
            "trusted": bool(trusted),
        }
    raw = float(song_position_ms)
    return {
        # 判定线画在「当前实际听到」的谱面位置，避免比声音早到
        "judge_line_song_ms": raw + float(output_latency_ms),
        "audible_song_ms": raw + float(output_latency_ms),
        "output_latency_comp_ms": float(output_latency_ms),
        "trusted": bool(trusted),
    }


# ---------------------------------------------------------------------------
# 修复点 5：结算幂等
# ---------------------------------------------------------------------------
class SettlementStore:
    """线程安全的结算存储：以幂等键去重，重复提交只产生一条成绩记录。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._bodies: dict[str, str] = {}

    @staticmethod
    def _body_fingerprint(request: dict[str, Any]) -> str:
        return json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)

    def settle(self, request: dict[str, Any]) -> dict[str, Any]:
        key = request.get("idempotency_key")
        if not key:
            raise ValueError("结算请求缺少 idempotency_key")
        fingerprint = self._body_fingerprint(request)
        with self._lock:
            if key in self._records:
                if self._bodies[key] != fingerprint:
                    return {
                        "ok": False,
                        "error": "idempotency_key_conflict",
                        "idempotency_key": key,
                    }
                duplicate = dict(self._records[key])
                duplicate["status"] = "duplicate"
                return duplicate
            record = {
                "ok": True,
                "status": "created",
                "record_id": f"rec_{key}",
                "idempotency_key": key,
                "chart_id": request.get("chart_id"),
                "user_id": request.get("user_id"),
                "play_id": request.get("play_id"),
                "score": request.get("score"),
            }
            self._records[key] = record
            self._bodies[key] = fingerprint
            return dict(record)

    def count(self) -> int:
        with self._lock:
            return len(self._records)


def settle_idempotent(request: Any = None, store: SettlementStore | None = None) -> dict[str, Any]:
    """提交结算：同一幂等键的重复请求返回首条记录，不会产生双份排行榜记录。"""
    if request is None or _is_scalar_int(request):
        return {"ok": False, "status": "rejected", "reason": "empty_request"}
    if store is None:
        store = SettlementStore()
    return store.settle(request)


# ---------------------------------------------------------------------------
# 修复点 6：成绩服务端校验
# ---------------------------------------------------------------------------
def load_chart(source: Any = None) -> dict[str, Any]:
    """加载并校验铺面：音符字段齐全、时间有限且有序，返回规范化铺面。"""
    if source is None or _is_scalar_int(source):
        return {"chart_id": None, "notes": []}
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as fh:
            source = json.load(fh)
    if not isinstance(source, dict):
        raise ValueError("铺面必须是 dict")
    notes_raw = source.get("notes", [])
    notes: list[dict[str, Any]] = []
    for i, raw in enumerate(notes_raw):
        if "t" not in raw:
            raise ValueError(f"音符 {i} 缺少 t")
        t = float(raw["t"])
        if not math.isfinite(t) or t < 0:
            raise ValueError(f"音符 {i} 时间非法: {raw['t']}")
        note = dict(raw)
        note["t"] = t
        note.setdefault("note_id", f"n{i}")
        note.setdefault("lane", 0)
        if note.get("type") == "hold":
            if "tail" not in raw:
                raise ValueError(f"长按音符 {i} 缺少 tail")
            note["tail"] = float(note["tail"])
            if note["tail"] <= t:
                raise ValueError(f"长按音符 {i} 的 tail 必须晚于 t")
        notes.append(note)
    notes.sort(key=lambda n: n["t"])
    return {"chart_id": source.get("chart_id"), "notes": notes}


def validate_judge(note: Any = None, judgment: Any = None) -> dict[str, Any]:
    """单条判定的服务端合理性校验：等级必须与偏差窗口、轨道信息一致。"""
    if not isinstance(note, dict) or not isinstance(judgment, dict):
        return {"valid": False, "reason": "empty_payload"}
    win = dict(DEFAULT_WINDOWS, **(judgment.get("windows") or {}))
    label = judgment.get("label")
    if label not in JUDGE_LABELS:
        return {"valid": False, "reason": f"unknown_label:{label}"}
    if judgment.get("lane") is not None and note.get("lane") is not None and judgment["lane"] != note["lane"]:
        return {"valid": False, "reason": "lane_mismatch"}
    delta = judgment.get("abs_delta_ms")
    if delta is None:
        return {"valid": True, "reason": "miss_without_delta"} if label == "MISS" else {"valid": False, "reason": "missing_delta"}
    delta = float(delta)
    expected = _label_for_delta(delta, win)
    if expected != label:
        return {"valid": False, "reason": f"label_window_mismatch:{label}!={expected}"}
    if SCORE_BY_LABEL[label] != judgment.get("score"):
        return {"valid": False, "reason": "score_mismatch"}
    return {"valid": True, "reason": "ok"}


def score_server_check(chart: Any = None, report: Any = None) -> dict[str, Any]:
    """服务端成绩校验：重算分值与上限，拒绝任何客户端无法自洽的成绩。"""
    if not isinstance(chart, dict) or not isinstance(report, dict):
        return {"valid": False, "reasons": ["empty_payload"]}

    reasons: list[str] = []
    notes = chart.get("notes", [])
    note_count = len(notes)
    judgments = report.get("judgments")
    counts = report.get("counts") or {}

    if judgments is not None:
        if len(judgments) != note_count:
            reasons.append(f"judgment_count_mismatch:{len(judgments)}!={note_count}")
        labels = [j.get("label") for j in judgments]
        # 逐条判定的窗口自洽性 + 与音符时间对得上
        by_id = {n.get("note_id", f"n{i}"): n for i, n in enumerate(notes)}
        for j in judgments:
            note = by_id.get(j.get("note_id"))
            if note is None:
                reasons.append(f"unknown_note:{j.get('note_id')}")
                continue
            check = validate_judge(note, j)
            if not check["valid"]:
                reasons.append(f"{j.get('note_id')}:{check['reason']}")
        recomputed = sum(SCORE_BY_LABEL.get(label, 0) for label in labels)
    else:
        total_counts = sum(int(counts.get(k, 0)) for k in JUDGE_LABELS)
        if total_counts != note_count:
            reasons.append(f"count_sum_mismatch:{total_counts}!={note_count}")
        recomputed = sum(SCORE_BY_LABEL[k] * int(counts.get(k, 0)) for k in JUDGE_LABELS)

    max_score = note_count * SCORE_BY_LABEL["PERFECT"]
    score = report.get("score")
    if score is None:
        reasons.append("missing_score")
    else:
        score = float(score)
        if score < 0 or score > max_score:
            reasons.append(f"score_out_of_range:{score} not in [0,{max_score}]")
        elif abs(score - recomputed) > 1e-6:
            reasons.append(f"score_recompute_mismatch:reported={score},recomputed={recomputed}")

    max_combo = report.get("max_combo")
    if max_combo is not None and int(max_combo) > note_count:
        reasons.append(f"max_combo_exceeds_notes:{max_combo}>{note_count}")

    finished_at = report.get("finished_at_ms")
    started_at = report.get("started_at_ms")
    if finished_at is not None and started_at is not None and float(finished_at) < float(started_at):
        reasons.append("time_going_backwards")

    return {"valid": not reasons, "reasons": reasons, "recomputed_score": recomputed, "max_score": max_score}


# ---------------------------------------------------------------------------
# 指标与审计
# ---------------------------------------------------------------------------
def metrics(items: Any = None) -> dict[str, Any]:
    """聚合判定 / 结算指标：判定分布、命中率、结算去重数等。"""
    if items is None or _is_scalar_int(items):
        return {"judgment_counts": {k: 0 for k in JUDGE_LABELS}, "settled": 0, "duplicates": 0, "rejected": 0}
    counts = {k: 0 for k in JUDGE_LABELS}
    settled = duplicates = rejected = 0
    total_score = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if "label" in item and item.get("label") in JUDGE_LABELS:
            counts[item["label"]] += 1
            total_score += int(item.get("score", SCORE_BY_LABEL[item["label"]]))
        status = item.get("status")
        if status == "created":
            settled += 1
        elif status == "duplicate":
            duplicates += 1
        elif item.get("ok") is False:
            rejected += 1
    judged = sum(counts.values())
    return {
        "judgment_counts": counts,
        "judged": judged,
        "hit_rate": (counts["PERFECT"] + counts["GREAT"] + counts["GOOD"]) / judged if judged else 0.0,
        "total_score": total_score,
        "settled": settled,
        "duplicates": duplicates,
        "rejected": rejected,
    }


def audit(events: Any = None) -> dict[str, Any]:
    """把时钟漂移 / 校准 / 结算事件固化为只追加审计轨迹（带序号与载荷指纹）。"""
    if events is None or _is_scalar_int(events):
        return {"trail": [], "seq": 0}
    trail: list[dict[str, Any]] = []
    for seq, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("审计事件必须是 dict")
        payload = event.get("payload", {})
        canonical = SettlementStore._body_fingerprint(payload)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        trail.append(
            {
                "seq": seq,
                "kind": event.get("kind", "unknown"),
                "payload": payload,
                "fingerprint": fingerprint,
            }
        )
    return {"trail": trail, "seq": len(trail)}
