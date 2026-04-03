DOCKER_COMPOSE ?= docker compose
DOCKER_SERVICE ?= apm

.PHONY: docker-build docker-shell docker-python-version docker-materialize-all docker-score docker-summarize docker-plot

docker-build:
	$(DOCKER_COMPOSE) build $(DOCKER_SERVICE)

docker-shell:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE)

docker-python-version:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) python --version

docker-materialize-all:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python -m apm.data.materialize_all \
		--project-root . \
		--config-dir configs/datasets \
		--sample-size 100 \
		--seed 42

docker-score:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python scratch/detector_scoring/run_detector_scores.py \
		--project-root . \
		--examples-per-label 3

docker-summarize:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python scratch/detector_scoring/summarize_scores.py \
		--project-root .

docker-plot:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python scratch/detector_scoring/plot_detector_scores.py \
		--project-root .
