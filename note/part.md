# SeqAvatar Part-MoE / State 相关笔记

更新时间：2026-07-12

## 本轮重点

- 2026-07-12 17:50 CST 读取并解释 `scripts/eval_state_subsets.py`：该脚本不是训练脚本，也不是常规全量指标评估脚本，而是用于 DNA-Rendering 上对 `state_warm_a04` 这类 state 方法做“局部/困难子集”诊断。它把 baseline `orginal` 的 novelview 渲染和指定 `state_experiment/state_run` 的 novelview 渲染逐图对齐，按高运动帧、mask 边界带、baseline 最大误差 crop 三类子集重新计算 L1/PSNR/SSIM/LPIPS，并输出 `subset_metrics.json` 和 `subset_metrics.md`，用于判断 state 方法是不是只在运动大、边界、局部高误差区域有收益或退化。

## 后续维护约定

- 每次继续讨论本项目时，先读本文件，再把新的结论、命令、实验结果或注意事项追加/更新到这里。
