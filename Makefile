.PHONY: setup setup-all test check inspect-tree

# third_party/ 不在版本控制里（三个上游各自带 .git，收录只会记成 gitlink；
# MyPCBench 的 qcow2 有 16.5G）。clone 本仓库之后先跑这个把它们建起来。
setup:
	bash scripts/setup_third_party.sh

# 再加 EvoCUA 和 OpenCUA-OSWorld，跑对应 agent 时才需要。
setup-all:
	bash scripts/setup_third_party.sh --all

test:
	PYTHONPATH=src python3 -m pytest -q tests

check:
	PYTHONPATH=src python3 -m derail.cli check-repository

inspect-tree:
	find configs schemas prompts src analysis infra patches scripts tests -maxdepth 3 -type f | sort
