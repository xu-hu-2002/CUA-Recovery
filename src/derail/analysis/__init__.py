"""论文分析模块。

分析代码必须读取结构化 results，不能从 LaTeX 表格反向抓取数字。
"""

from .grouping import group_by_agent_and_depth

__all__ = ["group_by_agent_and_depth"]
