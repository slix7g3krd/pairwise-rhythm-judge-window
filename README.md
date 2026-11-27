# pairwise-rhythm-judge-window

## 领域

游戏、图形与多媒体

## 主题

音游判定窗口

## 说明

统一音频时钟、事件时间戳与幂等结算。

初始状态下 `gameplay/rhythm_judge_window.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 故障与修复对应关系

| # | 故障现象 | 修复 | 主要入口 |
|---|----------|------|----------|
| 1 | 判定线按当前帧时间戳判定，音频时钟与渲染时钟各自漂移，长曲后偏移十几毫秒，Perfect 降 Great | 统一音频时钟：渲染帧只采样 `MasterClock`，音频回调上报实际播放位置并 EMA 修正漂移；音频断流走降级 | `MasterClock`、`unified_audio_clock`、`fallback`、`present` |
| 2 | 输入事件在系统队列排队，掉帧时多个点击同一帧到达，按到达顺序判定导致连打断连 | 按事件时间戳判定：事件统一换算到谱面时间域，按 `(event_ts, seq)` 规范排序后再匹配音符，与到达帧/入列顺序无关 | `determinism`、`judge_by_event_time`、`judge` |
| 3 | 长按尾判按松手事件处理，提前松手仍按满判定，分数虚高 | 尾判按区间：依据松手时刻相对尾判的窗口偏差评级，并按 `[head, tail]` 内按住区间并集计算覆盖率，提前松手判 MISS | `hold_tail_by_interval`、`hold` |
| 4 | 铺面校准值按设备写死，蓝牙耳机约 200ms 延迟使判定线比声音早到 | 延迟实测校准：多次实测取中位数，`cal = -(output + input)`；无样本/样本过期/换设备时要求重新校准，禁止回退写死值 | `latency_measured_calib` |
| 5 | 结算在异步线程算分，请求重复提交，同一曲目上传两次成绩 | 结算幂等：以 `idempotency_key` 去重（带锁 + 载荷指纹），重复请求返回首条记录，同键不同体判冲突 | `SettlementStore`、`settle_idempotent` |
| 6 | 排行榜出现双份/篡改记录 | 成绩服务端校验：服务端按音符数与判定窗口重算分值，校验分数上限、判定-偏差自洽、轨道、时间方向 | `score_server_check`、`validate_judge`、`load_chart` |

辅助入口：`metrics`（判定/结算聚合指标）、`audit`（带 sha256 指纹的只追加审计轨迹）。

## 测试

* `tests/test_rhythm_judge_window.py`：15 个入口的契约测试。
* `tests/test_fix_scenarios.py`：六个修复点的场景测试（同帧乱序、提前松手、蓝牙 200ms、并发重复结算、服务端改分拦截等），共 43 项。
