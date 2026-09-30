"""DERAIL benchmark construction and evaluation toolkit.

这个 package 只承载可复现的 benchmark/experiment 逻辑。论文文字、人工判断和
未冻结的实验假设不能偷偷硬编码进这里；它们应分别进入 prompts、configs 和
带 provenance 的 annotation records。
"""

__version__ = "0.1.0"
