# Modern Control Theory — Notes & Labs

这个仓库用于记录我学习现代控制理论过程中的笔记、课程作业/论文复现与小型实验（MATLAB / Python）。目标是把“理论 → 推导 → 复现 → 仿真结果”放在同一个可追溯的仓库里，方便在简历/面试中展示。

## 快速导航
- [`Cart-Pole System/`](<Cart-Pole System/>)：倒立摆建模/线性化 + LQR、系统辨识 + LQR、强化学习控制（PPO）
- [`23_wu_zhou1_amm/`](23_wu_zhou1_amm/)：事件触发控制相关工作/复现（MATLAB + YALMIP）
- [`20_zhou_liu_arik1_nn/`](20_zhou_liu_arik1_nn/)：基于 LMI 的稳定性可行性验证（Python + CVXPY）
- [`notes/`](notes/)：学习笔记（HTML/PDF）

## 推荐入口（适合 GitHub 直接阅读）
- 项目总览：[`Cart-Pole System/README.md`](<Cart-Pole System/README.md>)
- 三种控制范式对比的研究路线：[`Cart-Pole System/工作.md`](<Cart-Pole System/工作.md>)
- 笔记索引：[`notes/README.md`](notes/README.md)

## 仓库结构（摘要）
- `Cart-Pole System/`：倒立摆实验（Lab1/Lab2/Lab3）
- `20_zhou_liu_arik1_nn/`：LMI/数值复现（CVXPY）
- `23_wu_zhou1_amm/`：LMI/数值复现（YALMIP）
- `notes/`：现代控制/鲁棒控制/采样数据控制等资料整理

## 结果截图（部分）
![](<Cart-Pole System/Lab2/model_validation_plot_corrected.png>)
![](<Cart-Pole System/Lab3/rl_training_curve.png>)
