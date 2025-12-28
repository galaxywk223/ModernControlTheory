# 20_zhou_liu_arik1_nn

这一部分主要是基于论文/作业中的条件，使用 Python + CVXPY 搭建 LMI 并做可行性验证与参数扫描。

推荐入口：
- `20_zhou_liu_arik1_nn/solve1.py`：定理/推论的 LMI 搭建，支持多种 `D` 的解读与求解器回退；包含 `sigma2` 的最大化搜索。
- `20_zhou_liu_arik1_nn/solve1_.py`：另一版实现（更偏“复现论文表格”的流程）。
