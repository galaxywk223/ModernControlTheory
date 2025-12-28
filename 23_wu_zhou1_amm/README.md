# 23_wu_zhou1_amm

这一部分聚焦事件触发控制/半马尔可夫切换系统（semi-MSSs）相关的 LMI 复现与求解（MATLAB + YALMIP）。

推荐入口：

- `23_wu_zhou1_amm/solve.m`：YALMIP 脚本，按 Theorem 2 / Example 1 的结构搭建 LMI 并重构控制增益。
- `23_wu_zhou1_amm/2/`：阶段式求解/仿真脚本（含路径与求解器设置）。
- `23_wu_zhou1_amm/想法.pdf`：过程记录/思路草稿。
