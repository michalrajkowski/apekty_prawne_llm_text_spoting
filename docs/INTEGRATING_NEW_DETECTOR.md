# Integrating New Detector

Use this checklist for every new detector integration.

## Required Files

1. Detector adapter class in `src/apm/detectors/adapters/<detector_name>.py`.
2. Detector config in `configs/detectors/<detector_name>.detector.json`.
3. Smoke runner in `src/apm/detectors/adapters/<detector_name>_smoke.py`.
4. Detector-special tests in `tests/detectors/test_<detector_name>.py`.

## Required Interface Contract

Each detector adapter must subclass `apm.detectors.base.AbstractDetector` and implement all methods exactly:

```python
@classmethod
def initialize(cls, config: DetectorConfig) -> Self: ...

def predict_single(self, text: str) -> float: ...

def predict_batch(self, texts: Sequence[str]) -> list[float]: ...

def delete(self) -> None: ...
```

## Behavioral Requirements

1. `predict_single` returns one `float` score for one input text.
2. `predict_batch` returns one `float` per input text, preserving input order.
3. `predict_single(text)` must be consistent with `predict_batch([text])[0]`.
4. `delete()` must release model/tokenizer references and free CUDA cache when CUDA is used.
5. If the detector uses stochastic generation/sampling, it must support seeded behavior (`seed` in config, default `42`).

## Config Requirements

Each detector config must define at least:

1. `detector_id`
2. model identifier fields used by adapter (for example `model_id`, `base_model_id`, `perturbation_model_id`)
3. `device`
4. `cache_dir`
5. `local_files_only`
6. `trust_remote_code`

## Submodule Requirements

If detector code comes from GitHub:

1. Add repo under `external/detectors/` as git submodule.
2. Ensure bootstrap works with one command via `scripts/init_submodules.sh` (`git submodule update --init --recursive`).

## Test Requirements

1. Detector tests must use marker `detector_special`.
2. Default test suite must not run detector-special tests.
3. Detector tests must cover:
- batch scoring shape/type,
- single vs batch consistency,
- cleanup path (`delete`).
