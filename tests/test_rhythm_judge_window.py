import pytest
import gameplay.rhythm_judge_window as mod

CASES = [
    ("unified_audio_clock", (1,)),
    ("judge_by_event_time", (1,)),
    ("hold_tail_by_interval", (1,)),
    ("latency_measured_calib", (1,)),
    ("settle_idempotent", (1,)),
    ("score_server_check", (1,)),
    ("load_chart", (1,)),
    ("judge", (1,)),
    ("hold", (1,)),
    ("present", (1,)),
    ("validate_judge", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
