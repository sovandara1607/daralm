# DaraLM — common developer/deployment commands.

.DEFAULT_GOAL := help

IMAGE       := daralm-api
TAG         := latest
CONTAINER   := daralm-api
PORT        := 8000

# Override these variables to serve another checkpoint.
CONFIG      := configs/50m.yaml
CHECKPOINT  := checkpoints/daralm-50m/best
TOKENIZER   := checkpoints/tokenizer/unigram.model

.PHONY: help install test lint format serve \
        docker-build docker-run docker-stop docker-logs docker-shell clean

help: ## Show this list of targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync all dependencies (incl. dev) into .venv via uv
	uv sync

test: ## Run the full pytest suite
	uv run pytest

lint: ## Run ruff check
	uv run ruff check .

format: ## Auto-format with ruff
	uv run ruff format .

serve: ## Run the API locally with auto-reload (uses configs/50m.yaml by default)
	uv run uvicorn api.main:app --reload

docker-build: ## Build the API image
	docker build -t $(IMAGE):$(TAG) .

docker-run: ## Run the image, mounting configs/checkpoints as read-only volumes
	@test -d $(CHECKPOINT) || { \
		echo "error: checkpoint dir '$(CHECKPOINT)' not found on the host."; \
		echo "Train one first (see README), or override CONFIG/CHECKPOINT/TOKENIZER, e.g.:"; \
		echo "  make docker-run CONFIG=configs/50m-instruct.yaml CHECKPOINT=checkpoints/daralm-50m-instruct/best"; \
		exit 1; \
	}
	docker run --rm -d \
		--name $(CONTAINER) \
		-p $(PORT):8000 \
		-e DARALM_CONFIG=/app/$(CONFIG) \
		-e DARALM_CHECKPOINT=/app/$(CHECKPOINT) \
		-e DARALM_TOKENIZER=/app/$(TOKENIZER) \
		-v $(abspath configs):/app/configs:ro \
		-v $(abspath checkpoints):/app/checkpoints:ro \
		$(IMAGE):$(TAG)
	@echo "Started. Follow logs with 'make docker-logs'; try 'curl http://localhost:$(PORT)/health'."

docker-stop: ## Stop the running container
	docker stop $(CONTAINER)

docker-logs: ## Tail the running container's logs
	docker logs -f $(CONTAINER)

docker-shell: ## Open a shell inside a throwaway container built from the image
	docker run --rm -it --entrypoint /bin/bash $(IMAGE):$(TAG)

clean: ## Remove Python/pytest/ruff caches
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
