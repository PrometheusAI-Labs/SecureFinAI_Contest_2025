# SecureFinAI Contest 2025 Task 3 - Makefile

.PHONY: help install test run clean

help: ## Show this help message
	@echo "SecureFinAI Contest 2025 Task 3 - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	uv sync

install-dev: ## Install development dependencies
	uv sync --extra dev

test: ## Run quick test
	python quick_test.py

run: ## Run main evaluation (10 samples)
	python main.py --max-samples 10 --compare-baseline

run-full: ## Run full evaluation (all samples)
	python main.py --max-samples 1000 --compare-baseline

run-spanish: ## Run evaluation on Spanish dataset
	python main.py --dataset TheFinAI/SecureFinAI_Contest_2025-Task_3_SpanishOCR --max-samples 10 --compare-baseline

clean: ## Clean output files
	rm -rf results/
	rm -rf preds_*/
	rm -f *.csv

setup: install test ## Setup environment and run test

all: setup run ## Setup, test, and run evaluation