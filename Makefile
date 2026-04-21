DOCKER_COMPOSE ?= docker compose
DOCKER_SERVICE ?= apm

.PHONY: docker-build docker-shell docker-python-version docker-materialize-all docker-materialize-splits docker-score docker-summarize docker-plot docker-experiment

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

docker-materialize-splits:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python -m apm.experiments.split_materialize \
		--project-root . \
		--datasets hc3:all_train grid:filtered \
		--train-ratio 0.7 \
		--seed 42

docker-score:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python scratch/detector_scoring/run_detector_scores.py \
		--project-root . \
		--examples-per-label 30 \
		--model-runs aigc_detector_env3 seqxgpt:gpt2_medium seqxgpt:gpt_j_6b

docker-summarize:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python scratch/detector_scoring/summarize_scores.py \
		--project-root .

docker-plot:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python scratch/detector_scoring/plot_detector_scores.py \
		--project-root .

docker-experiment:
	$(DOCKER_COMPOSE) run --rm $(DOCKER_SERVICE) \
		python -m apm.experiments.runner \
		--project-root . \
		--datasets hc3:all_train grid:filtered \
		--model-runs aigc_detector_env3 seqxgpt:gpt2_medium seqxgpt:gpt_j_6b \
		--train-examples-per-label 100 \
		--evaluation-examples-per-label 100 \
		--threshold-objective balanced_accuracy \
		--seed 42
