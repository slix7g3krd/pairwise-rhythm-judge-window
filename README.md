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
