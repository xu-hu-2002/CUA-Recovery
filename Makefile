.PHONY: setup setup-all check inspect-tree

setup:
	bash scripts/setup_third_party.sh

setup-all:
	bash scripts/setup_third_party.sh --all

check:
	PYTHONPATH=src python3 -m recovery.cli check-repository

inspect-tree:
	find configs schemas prompts src analysis infra patches scripts -maxdepth 3 -type f | sort
