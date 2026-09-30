.PHONY: setup setup-all test check inspect-tree

setup:
	bash scripts/setup_third_party.sh

setup-all:
	bash scripts/setup_third_party.sh --all

test:
	PYTHONPATH=src python3 -m pytest -q tests

check:
	PYTHONPATH=src python3 -m derail.cli check-repository

inspect-tree:
	find configs schemas prompts src analysis infra patches scripts tests -maxdepth 3 -type f | sort
