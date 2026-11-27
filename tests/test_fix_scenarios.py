"""六个修复点的场景测试：复现故障输入，断言修复后的正确行为。"""

import threading

import pytest

from gameplay.rhythm_judge_window import (
    DEFAULT_WINDOWS,
    MasterClock,
    SettlementStore,
    audit,
    determinism,
    fallback,
    hold,
    hold_tail_by_interval,
    judge,
    judge_by_event_time,
    latency_measured_calib,
    load_chart,
    metrics,
    present,
    score_server_check,
    settle_idempotent,
    unified_audio_clock,
    validate_judge,
)


# ---------------------------------------------------------------------------
# 修复点 1：统一音频时钟 —— 渲染时钟不能独立漂移
# ---------------------------------------------------------------------------
class TestUnifiedAudioClock:
    def test_render_samples_single_clock_and_drift_is_corrected(self):
        clock = MasterClock()
        clock.anchor(audio_hw_ms=1000.0, song_ms=0.0)

        # 无漂移时，渲染帧采样值等于外推谱面时间
        assert clock.song_ms(2000.0) == pytest.approx(1000.0)

        # 音频硬件实际位置比外推慢了 15ms（长曲漂移），回调持续上报后被 EMA 收敛
        for hw in (1100.0, 1200.0, 1300.0):
            expected_extrapolated = hw - 1000.0
            clock.update_audio_position(hw, expected_extrapolated - 15.0)
        assert clock.drift_ms < 0
        assert clock.song_ms(2000.0) == pytest.approx(1000.0 + clock.drift_ms)

    def test_pure_clock_matches_master_clock_formula(self):
        # 渲染侧没有任何独立时间源，只做锚点 + 漂移修正的线性换算
        assert unified_audio_clock(2500.0, anchor_hw_ms=1000.0, anchor_song_ms=500.0, drift_ms=-12.0) == \
            pytest.approx(1988.0)

    def test_stall_fallback_marks_untrusted(self):
        clock = MasterClock()
        clock.anchor(0.0, 0.0)
        clock.update_audio_position(100.0, 100.0)
        state = fallback(clock)
        assert state["trusted"] is False
        assert state["mode"] == "fallback"
        assert clock.stalled is True


# ---------------------------------------------------------------------------
# 修复点 2：按事件时间戳判定 —— 同帧批量到达也要按按键时刻排序
# ---------------------------------------------------------------------------
class TestJudgeByEventTime:
    def test_same_frame_burst_orders_by_event_timestamp(self):
        # 两个点击在同一渲染帧到达（arrival_frame 相同），但事件时间戳有先后
        notes = [
            {"note_id": "n0", "t": 1000.0, "lane": 0},
            {"note_id": "n1", "t": 1010.0, "lane": 0},
        ]
        events = [
            {"event_id": "e1", "event_ts": 1011.0, "arrival_frame": 50, "lane": 0, "seq": 1},
            {"event_id": "e0", "event_ts": 1000.0, "arrival_frame": 50, "lane": 0, "seq": 0},
        ]
        result = judge_by_event_time(notes, events, now_ms=2000.0)
        judgments = result["judgments"]
        # 判定顺序由 event_ts 决定，与入列顺序、到达帧无关
        assert [j["event_id"] for j in judgments] == ["e0", "e1"]
        assert judgments[0]["note_id"] == "n0"
        assert judgments[0]["label"] == "PERFECT"
        assert judgments[1]["note_id"] == "n1"
        # 连打段连击不被乱序打断
        assert [j["combo"] for j in judgments] == [1, 2]
        assert result["max_combo"] == 2

    def test_shuffled_delivery_is_deterministic(self):
        events = [
            {"event_id": "e2", "event_ts": 300.0, "seq": 2},
            {"event_id": "e0", "event_ts": 100.0, "seq": 0},
            {"event_id": "e1", "event_ts": 200.0, "seq": 1},
        ]
        forward = determinism(events)
        backward = determinism(list(reversed(events)))
        assert forward["ordered_event_ids"] == ["e0", "e1", "e2"]
        assert forward["shuffle_stable"] is True
        assert backward["ordered_event_ids"] == forward["ordered_event_ids"]

    def test_unhit_note_past_window_is_miss(self):
        notes = [{"note_id": "n0", "t": 500.0, "lane": 0}]
        result = judge_by_event_time(notes, [], now_ms=10_000.0)
        assert result["judgments"][0]["label"] == "MISS"
        assert result["judgments"][0]["combo"] == 0

    def test_eleven_ms_drift_downgrades_without_unified_clock_fix_is_calibrated(self):
        # 原故障：长曲漂移 11ms 把 PERFECT 判成 GREAT；统一时钟 + 漂移修正后偏差归零
        assert judge(1000.0, 1011.0, cal_ms=0.0)["label"] == "PERFECT"  # 40ms 窗口内仍是 PERFECT
        corrected = judge(1000.0, 1011.0, cal_ms=-11.0)
        assert corrected["delta_ms"] == pytest.approx(0.0)
        assert corrected["label"] == "PERFECT"


# ---------------------------------------------------------------------------
# 修复点 3：尾判按区间 —— 提前松手不能按满判定
# ---------------------------------------------------------------------------
class TestHoldTailByInterval:
    def test_early_release_is_miss_not_full_score(self):
        result = hold_tail_by_interval(
            tail_time=2000.0,
            release_time=1700.0,  # 提前 300ms，远超 GOOD 窗口
            held_intervals=[(1000.0, 1700.0)],
            head_time=1000.0,
        )
        assert result["label"] == "MISS"
        assert result["score"] == 0
        assert result["early_release"] is True

    def test_release_inside_tail_window_graded_by_distance(self):
        perfect = hold_tail_by_interval(
            tail_time=2000.0, release_time=2010.0,
            held_intervals=[(1000.0, 2010.0)], head_time=1000.0,
        )
        great = hold_tail_by_interval(
            tail_time=2000.0, release_time=2050.0,
            held_intervals=[(1000.0, 2050.0)], head_time=1000.0,
        )
        assert perfect["label"] == "PERFECT"
        assert great["label"] == "GREAT"

    def test_coverage_reflects_gap_during_hold(self):
        # 中途松手 200ms：覆盖率 = 800/1000
        result = hold_tail_by_interval(
            tail_time=2000.0,
            release_time=2005.0,
            held_intervals=[(1000.0, 1400.0), (1600.0, 2005.0)],
            head_time=1000.0,
        )
        assert result["coverage_ratio"] == pytest.approx(0.8)

    def test_full_hold_flow_early_release_cannot_inflate_score(self):
        note = {"note_id": "h0", "t": 1000.0, "tail": 2000.0, "lane": 1, "type": "hold"}
        events = [
            {"event_id": "d", "event_ts": 1000.0, "lane": 1, "kind": "down", "seq": 0},
            {"event_id": "u", "event_ts": 1700.0, "lane": 1, "kind": "up", "seq": 1},
        ]
        result = hold(note, events)
        assert result["tail"]["label"] == "MISS"
        assert result["label"] == "MISS"
        # 头判满分也不能把整条长按抬成满分
        assert result["score"] < 2000


# ---------------------------------------------------------------------------
# 修复点 4：延迟实测校准 —— 蓝牙 200ms 不能再用写死值
# ---------------------------------------------------------------------------
class TestLatencyCalibration:
    def test_bt_headphone_measured_latency_shifts_calibration(self):
        cal = latency_measured_calib([198.0, 202.0, 200.0], device_id="bt-earbuds")
        assert cal["output_latency_ms"] == pytest.approx(200.0)
        assert cal["cal_ms"] == pytest.approx(-200.0)
        assert cal["needs_calibration"] is False

        # 校准值进判定：玩家物理上按声音拍点（事件晚 200ms），校准后偏差归零
        verdict = judge(1000.0, 1200.0, cal_ms=cal["cal_ms"])
        assert verdict["delta_ms"] == pytest.approx(0.0)
        assert verdict["label"] == "PERFECT"

    def test_missing_measurement_forbids_hardcoded_fallback(self):
        cal = latency_measured_calib(None, device_id="new-usb-dac")
        assert cal["needs_calibration"] is True
        assert cal["reason"] == "no_measurement"

    def test_stale_samples_after_device_switch_require_recalibration(self):
        cal = latency_measured_calib({"samples": [50.0], "stale": True, "device_id": "speaker"})
        assert cal["needs_calibration"] is True
        assert cal["reason"] == "stale_samples"

    def test_present_uses_same_latency_compensation_as_judge(self):
        frame = present(song_position_ms=800.0, output_latency_ms=200.0)
        # 判定线画在实际听到的位置，不再比声音早到 200ms
        assert frame["judge_line_song_ms"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# 修复点 5：结算幂等 —— 重复提交只能有一条成绩
# ---------------------------------------------------------------------------
class TestSettlementIdempotency:
    def _request(self, key="k-1", score=9500):
        return {
            "idempotency_key": key,
            "chart_id": "c1",
            "user_id": "u1",
            "play_id": "p1",
            "score": score,
        }

    def test_duplicate_submission_returns_same_record_once(self):
        store = SettlementStore()
        first = settle_idempotent(self._request(), store)
        second = settle_idempotent(self._request(), store)
        third = settle_idempotent(self._request(), store)
        assert first["status"] == "created"
        assert second["status"] == "duplicate"
        assert third["status"] == "duplicate"
        assert first["record_id"] == second["record_id"] == third["record_id"]
        assert store.count() == 1  # 排行榜不会出现双份记录

    def test_same_key_different_body_is_conflict(self):
        store = SettlementStore()
        settle_idempotent(self._request(score=9500), store)
        conflict = settle_idempotent(self._request(score=10_000), store)
        assert conflict["ok"] is False
        assert conflict["error"] == "idempotency_key_conflict"
        assert store.count() == 1

    def test_concurrent_duplicate_requests_create_single_record(self):
        store = SettlementStore()
        outcomes: list[dict] = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            outcomes.append(settle_idempotent(self._request(), store))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert store.count() == 1
        assert sum(1 for o in outcomes if o["status"] == "created") == 1
        assert sum(1 for o in outcomes if o["status"] == "duplicate") == 7


# ---------------------------------------------------------------------------
# 修复点 6：成绩服务端校验 —— 客户端改分必须被拦下
# ---------------------------------------------------------------------------
class TestScoreServerCheck:
    def _chart(self):
        return {
            "chart_id": "c1",
            "notes": [
                {"note_id": "n0", "t": 100.0, "lane": 0},
                {"note_id": "n1", "t": 200.0, "lane": 0},
            ],
        }

    def test_honest_full_perfect_report_passes(self):
        report = {
            "score": 2000,
            "max_combo": 2,
            "started_at_ms": 0,
            "finished_at_ms": 999,
            "judgments": [
                {"note_id": "n0", "label": "PERFECT", "abs_delta_ms": 5.0, "score": 1000, "lane": 0},
                {"note_id": "n1", "label": "PERFECT", "abs_delta_ms": 8.0, "score": 1000, "lane": 0},
            ],
        }
        verdict = score_server_check(self._chart(), report)
        assert verdict["valid"] is True
        assert verdict["recomputed_score"] == 2000

    def test_inflated_score_is_rejected(self):
        report = {
            "score": 999_999,  # 超过上限且与重算不符
            "max_combo": 2,
            "judgments": [
                {"note_id": "n0", "label": "PERFECT", "abs_delta_ms": 5.0, "score": 1000, "lane": 0},
                {"note_id": "n1", "label": "GREAT", "abs_delta_ms": 50.0, "score": 700, "lane": 0},
            ],
        }
        verdict = score_server_check(self._chart(), report)
        assert verdict["valid"] is False
        assert any("score_recompute_mismatch" in r or "score_out_of_range" in r for r in verdict["reasons"])
        assert verdict["recomputed_score"] == 1700

    def test_judgment_window_contradiction_is_rejected(self):
        # 偏差 100ms 只能是 GOOD，却声称 PERFECT
        report = {
            "score": 2000,
            "judgments": [
                {"note_id": "n0", "label": "PERFECT", "abs_delta_ms": 100.0, "score": 1000, "lane": 0},
                {"note_id": "n1", "label": "PERFECT", "abs_delta_ms": 8.0, "score": 1000, "lane": 0},
            ],
        }
        verdict = score_server_check(self._chart(), report)
        assert verdict["valid"] is False
        assert any("label_window_mismatch" in r for r in verdict["reasons"])

    def test_count_based_report_with_wrong_sum_rejected(self):
        report = {"score": 2000, "counts": {"PERFECT": 3, "GREAT": 0, "GOOD": 0, "MISS": 0}}
        verdict = score_server_check(self._chart(), report)
        assert verdict["valid"] is False
        assert any("count_sum_mismatch" in r for r in verdict["reasons"])

    def test_lane_mismatch_rejected_by_judge_validator(self):
        note = {"note_id": "n0", "t": 100.0, "lane": 0}
        verdict = validate_judge(note, {"label": "PERFECT", "abs_delta_ms": 5.0, "score": 1000, "lane": 2})
        assert verdict["valid"] is False


# ---------------------------------------------------------------------------
# 铺面加载与辅助入口
# ---------------------------------------------------------------------------
class TestChartAndHelpers:
    def test_load_chart_sorts_and_validates(self):
        chart = load_chart(
            {
                "chart_id": "c",
                "notes": [
                    {"note_id": "b", "t": 300.0},
                    {"note_id": "a", "t": 100.0},
                    {"note_id": "h", "t": 50.0, "type": "hold", "tail": 250.0},
                ],
            }
        )
        assert [n["note_id"] for n in chart["notes"]] == ["h", "a", "b"]

    def test_load_chart_rejects_bad_hold(self):
        with pytest.raises(ValueError):
            load_chart({"notes": [{"t": 100.0, "type": "hold", "tail": 50.0}]})

    def test_metrics_aggregation(self):
        summary = metrics(
            [
                {"label": "PERFECT", "score": 1000},
                {"label": "GREAT", "score": 700},
                {"label": "MISS", "score": 0},
                {"status": "created"},
                {"status": "duplicate"},
            ]
        )
        assert summary["judgment_counts"]["PERFECT"] == 1
        assert summary["hit_rate"] == pytest.approx(2 / 3)
        assert summary["settled"] == 1
        assert summary["duplicates"] == 1

    def test_audit_trail_is_sequential_and_fingerprinted(self):
        trail = audit([{"kind": "settle", "payload": {"a": 1}}, {"kind": "calib", "payload": {"x": 2}}])
        assert trail["seq"] == 2
        assert [e["seq"] for e in trail["trail"]] == [0, 1]
        assert len(trail["trail"][0]["fingerprint"]) == 64  # sha256 hex

    def test_windows_constants_are_ordered(self):
        assert DEFAULT_WINDOWS["perfect"] < DEFAULT_WINDOWS["great"] < DEFAULT_WINDOWS["good"]
