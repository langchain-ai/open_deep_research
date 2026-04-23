SHELL := /bin/bash

.PHONY: bootstrap venv dev llamator-dry llamator-live verify-nonleak

bootstrap:
	bash bootstrap.sh

venv:
	python3 -m venv .venv || true
	source .venv/bin/activate && python -m pip install -U pip wheel
	source .venv/bin/activate && python -m pip install -e .
	source .venv/bin/activate && python -m pip install -U python-dotenv

dev: venv
	source .venv/bin/activate && langgraph dev

LLAMATOR_VENV := .venv-llamator

.PHONY: llamator-venv llamator-dry llamator-live

llamator-venv:
	python3 -m venv $(LLAMATOR_VENV) || true
	source $(LLAMATOR_VENV)/bin/activate && python -m pip install -U pip wheel
	source $(LLAMATOR_VENV)/bin/activate && python -m pip install -U llamator

llamator-dry: llamator-venv
	# LIVE_API=0 by default => dry/skip
	source $(LLAMATOR_VENV)/bin/activate && python security/llamator/run_llamator.py

llamator-live: llamator-venv
	# Requires LIVE_API=1 and valid keys in .env
	source $(LLAMATOR_VENV)/bin/activate && python security/llamator/run_llamator.py


verify-nonleak:
	bash scripts/non_leak_check.sh
