"""HC3 dataset adapter implemented with the Hugging Face `datasets` library."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, cast

from datasets import Dataset, load_dataset

from apm.types import CanonicalDatasetRecord, DatasetLabel


class SourceDatasetLike(Protocol):
    """Minimal dataset protocol needed by the HC3 adapter."""

    def __len__(self) -> int:
        """Return number of source rows in the dataset split."""

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Iterate over source rows as mapping-like objects."""

    def select(self, indices: Sequence[int]) -> "SourceDatasetLike":
        """Return selected subset of rows by absolute indices."""


SourceDatasetLoader = Callable[[str, str, str], SourceDatasetLike]


@dataclass(frozen=True, slots=True)
class Hc3Selector:
    """Source selector for one HC3 subset/config and split."""

    config: str
    split: str


@dataclass(frozen=True, slots=True)
class Hc3Mapping:
    """Field-level mapping from HC3 source rows into canonical records."""

    id_field: str
    prompt_field: str
    human_answers_field: str
    ai_answers_field: str
    optional_source_field: str | None
    explode_answer_lists: bool
    canonical_label_mapping: dict[str, DatasetLabel]
    sample_id_pattern: str


@dataclass(frozen=True, slots=True)
class Hc3Config:
    """Parsed config for the HC3 adapter."""

    dataset_id: str
    source_uri: str
    default_selector: Hc3Selector
    selectors: dict[str, Hc3Selector]
    mapping: Hc3Mapping


@dataclass(frozen=True, slots=True)
class Hc3SelectorValidationSummary:
    """Validation summary for one selector load-check run."""

    selector: str
    dataset_rows: int
    loaded_rows: int
    missing_required_rows: int
    empty_human_answers_rows: int
    empty_ai_answers_rows: int


@dataclass(frozen=True, slots=True)
class Hc3LoadCheckReport:
    """Result of checking multiple HC3 selectors with fixed row count."""

    dataset_id: str
    source_uri: str
    requested_rows_per_selector: int
    summaries: tuple[Hc3SelectorValidationSummary, ...]


def _load_source_dataset(source_uri: str, config: str, split: str) -> SourceDatasetLike:
    """Load one HF dataset split with the official datasets library."""

    loaded = load_dataset(path=source_uri, name=config, split=split)
    if not isinstance(loaded, Dataset):
        raise ValueError("Expected `datasets.Dataset` when loading a concrete split.")
    return cast(SourceDatasetLike, loaded)


def _parse_selector(raw_selector: Mapping[str, Any], field_name: str) -> Hc3Selector:
    """Parse selector definition from config mapping."""

    config_value = raw_selector.get("config")
    split_value = raw_selector.get("split")
    if not isinstance(config_value, str) or not config_value.strip():
        raise ValueError(f"{field_name}.config must be a non-empty string.")
    if not isinstance(split_value, str) or not split_value.strip():
        raise ValueError(f"{field_name}.split must be a non-empty string.")
    return Hc3Selector(config=config_value, split=split_value)


def _parse_label(value: Any, field_name: str) -> DatasetLabel:
    """Parse and validate canonical dataset label."""

    if value not in {"human", "ai"}:
        raise ValueError(f"{field_name} must be one of ['human', 'ai'].")
    return cast(DatasetLabel, value)


def load_hc3_config(config_path: Path) -> Hc3Config:
    """Load HC3 adapter config from JSON file."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HC3 config root must be a JSON object.")

    dataset_id = payload.get("dataset_id")
    source_uri = payload.get("source_uri")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise ValueError("source_uri must be a non-empty string.")

    raw_default_selector = payload.get("default_selector")
    if not isinstance(raw_default_selector, dict):
        raise ValueError("default_selector must be an object.")
    default_selector = _parse_selector(raw_default_selector, "default_selector")

    raw_selectors = payload.get("selectors")
    if not isinstance(raw_selectors, dict) or not raw_selectors:
        raise ValueError("selectors must be a non-empty object.")
    selectors: dict[str, Hc3Selector] = {}
    for selector_name, raw_selector in raw_selectors.items():
        if not isinstance(selector_name, str) or not selector_name.strip():
            raise ValueError("Selector key must be a non-empty string.")
        if not isinstance(raw_selector, dict):
            raise ValueError(f"Selector {selector_name!r} must be an object.")
        selectors[selector_name] = _parse_selector(raw_selector, f"selectors[{selector_name!r}]")

    raw_mapping = payload.get("mapping")
    if not isinstance(raw_mapping, dict):
        raise ValueError("mapping must be an object.")

    id_field = raw_mapping.get("id_field")
    prompt_field = raw_mapping.get("prompt_field")
    human_answers_field = raw_mapping.get("human_answers_field")
    ai_answers_field = raw_mapping.get("ai_answers_field")
    optional_source_field = raw_mapping.get("optional_source_field")
    explode_answer_lists = raw_mapping.get("explode_answer_lists")
    sample_id_pattern = raw_mapping.get("sample_id_pattern")
    if not isinstance(id_field, str) or not id_field.strip():
        raise ValueError("mapping.id_field must be a non-empty string.")
    if not isinstance(prompt_field, str) or not prompt_field.strip():
        raise ValueError("mapping.prompt_field must be a non-empty string.")
    if not isinstance(human_answers_field, str) or not human_answers_field.strip():
        raise ValueError("mapping.human_answers_field must be a non-empty string.")
    if not isinstance(ai_answers_field, str) or not ai_answers_field.strip():
        raise ValueError("mapping.ai_answers_field must be a non-empty string.")
    if optional_source_field is not None and not isinstance(optional_source_field, str):
        raise ValueError("mapping.optional_source_field must be a string or null.")
    if not isinstance(explode_answer_lists, bool):
        raise ValueError("mapping.explode_answer_lists must be boolean.")
    if not isinstance(sample_id_pattern, str) or not sample_id_pattern.strip():
        raise ValueError("mapping.sample_id_pattern must be a non-empty string.")

    raw_label_mapping = raw_mapping.get("canonical_label_mapping")
    if not isinstance(raw_label_mapping, dict):
        raise ValueError("mapping.canonical_label_mapping must be an object.")

    canonical_label_mapping: dict[str, DatasetLabel] = {}
    for source_field_name in [human_answers_field, ai_answers_field]:
        canonical_label_mapping[source_field_name] = _parse_label(
            raw_label_mapping.get(source_field_name),
            f"mapping.canonical_label_mapping[{source_field_name!r}]",
        )

    mapping = Hc3Mapping(
        id_field=id_field,
        prompt_field=prompt_field,
        human_answers_field=human_answers_field,
        ai_answers_field=ai_answers_field,
        optional_source_field=optional_source_field,
        explode_answer_lists=explode_answer_lists,
        canonical_label_mapping=canonical_label_mapping,
        sample_id_pattern=sample_id_pattern,
    )

    return Hc3Config(
        dataset_id=dataset_id,
        source_uri=source_uri,
        default_selector=default_selector,
        selectors=selectors,
        mapping=mapping,
    )


class HC3Adapter:
    """Dataset adapter that maps HC3 source rows into canonical records."""

    def __init__(self, config: Hc3Config, dataset_loader: SourceDatasetLoader = _load_source_dataset) -> None:
        self._config = config
        self._dataset_loader = dataset_loader
        self.dataset_id = config.dataset_id

    @classmethod
    def from_config_path(cls, config_path: Path) -> "HC3Adapter":
        """Create adapter from config file path."""

        return cls(config=load_hc3_config(config_path))

    @property
    def source_uri(self) -> str:
        """Return configured source dataset URI."""

        return self._config.source_uri

    def list_splits(self) -> tuple[str, ...]:
        """List selector names that act as logical splits."""

        return tuple(self._config.selectors.keys())

    def _resolve_selector(self, split: str) -> Hc3Selector:
        """Resolve selector by logical split name."""

        if split not in self._config.selectors:
            raise ValueError(f"Unknown HC3 selector {split!r}.")
        return self._config.selectors[split]

    def _load_selector_dataset(self, selector: Hc3Selector) -> SourceDatasetLike:
        """Load the selected HF dataset split using configured loader."""

        return self._dataset_loader(self._config.source_uri, selector.config, selector.split)

    def _row_texts(self, row: Mapping[str, Any], source_field: str) -> list[str]:
        """Extract non-empty answer texts from one mapped list field."""

        raw_values = row.get(source_field)
        if not isinstance(raw_values, list):
            raise ValueError(f"Row field {source_field!r} must be a list.")
        texts: list[str] = []
        for value in raw_values:
            if isinstance(value, str) and value.strip():
                texts.append(value)
        return texts

    def _build_records_from_source_row(
        self,
        row_payload: Mapping[str, Any],
        row_index: int,
        selector_name: str,
        selector: Hc3Selector,
    ) -> list[CanonicalDatasetRecord]:
        """Convert one HC3 source row to canonical records."""

        mapping = self._config.mapping
        row_id_value = row_payload.get(mapping.id_field)
        prompt_value = row_payload.get(mapping.prompt_field)
        if not isinstance(row_id_value, str) or not row_id_value.strip():
            raise ValueError("HC3 row id must be a non-empty string.")
        if not isinstance(prompt_value, str):
            raise ValueError("HC3 question field must be a string.")

        optional_source_value: str | None = None
        if mapping.optional_source_field is not None:
            raw_source = row_payload.get(mapping.optional_source_field)
            if isinstance(raw_source, str):
                optional_source_value = raw_source

        source_to_texts: list[tuple[str, list[str]]] = [
            (mapping.human_answers_field, self._row_texts(row_payload, mapping.human_answers_field)),
            (mapping.ai_answers_field, self._row_texts(row_payload, mapping.ai_answers_field)),
        ]

        records: list[CanonicalDatasetRecord] = []
        for source_field_name, texts in source_to_texts:
            label = mapping.canonical_label_mapping[source_field_name]
            if mapping.explode_answer_lists:
                answer_entries = [(index, text_value) for index, text_value in enumerate(texts)]
            else:
                merged = "\n\n".join(texts).strip()
                answer_entries = [(0, merged)] if merged else []

            for answer_index, text_value in answer_entries:
                sample_id = mapping.sample_id_pattern.format(
                    config=selector.config,
                    split=selector.split,
                    id=row_id_value,
                    label=label,
                    answer_index=answer_index,
                )
                source_fields: dict[str, Any] = {
                    "source_dataset": self._config.source_uri,
                    "source_config": selector.config,
                    "source_split": selector.split,
                    "source_id": row_id_value,
                    "row_index": row_index,
                    "question": prompt_value,
                    "answer_field": source_field_name,
                    "answer_index": answer_index,
                }
                if optional_source_value is not None:
                    source_fields["source"] = optional_source_value

                records.append(
                    CanonicalDatasetRecord(
                        dataset_id=self.dataset_id,
                        split=selector_name,
                        sample_id=sample_id,
                        text=text_value,
                        label=label,
                        source_fields=source_fields,
                    )
                )
        return records

    def _load_records_from_dataset(
        self,
        dataset_rows: SourceDatasetLike,
        selector_name: str,
        selector: Hc3Selector,
    ) -> list[CanonicalDatasetRecord]:
        """Transform loaded source dataset rows into canonical record list."""

        records: list[CanonicalDatasetRecord] = []
        for row_index, source_row in enumerate(dataset_rows):
            records.extend(
                self._build_records_from_source_row(
                    row_payload=source_row,
                    row_index=row_index,
                    selector_name=selector_name,
                    selector=selector,
                )
            )
        return records

    def load_split(self, split: str) -> list[CanonicalDatasetRecord]:
        """Load one selector and map all source rows into canonical records."""

        selector = self._resolve_selector(split)
        source_dataset = self._load_selector_dataset(selector)
        return self._load_records_from_dataset(source_dataset, selector_name=split, selector=selector)

    def load_split_head(self, split: str, source_rows_limit: int) -> list[CanonicalDatasetRecord]:
        """Load at most `source_rows_limit` source rows for one selector."""

        if source_rows_limit <= 0:
            raise ValueError("source_rows_limit must be > 0.")

        selector = self._resolve_selector(split)
        source_dataset = self._load_selector_dataset(selector)
        selected_size = min(source_rows_limit, len(source_dataset))
        selected_rows = source_dataset.select(list(range(selected_size)))
        return self._load_records_from_dataset(selected_rows, selector_name=split, selector=selector)

    def validate_selector_rows(self, split: str, sample_rows: int) -> Hc3SelectorValidationSummary:
        """Validate first `sample_rows` source rows for one selector."""

        if sample_rows <= 0:
            raise ValueError("sample_rows must be > 0.")

        selector = self._resolve_selector(split)
        source_dataset = self._load_selector_dataset(selector)
        dataset_rows = len(source_dataset)
        selected_size = min(sample_rows, dataset_rows)
        selected_rows = source_dataset.select(list(range(selected_size)))

        missing_required_rows = 0
        empty_human_answers_rows = 0
        empty_ai_answers_rows = 0
        mapping = self._config.mapping

        for row_payload in selected_rows:
            required_fields = {
                mapping.id_field,
                mapping.prompt_field,
                mapping.human_answers_field,
                mapping.ai_answers_field,
            }
            if not required_fields.issubset(row_payload.keys()):
                missing_required_rows += 1
                continue

            human_answers = row_payload.get(mapping.human_answers_field)
            ai_answers = row_payload.get(mapping.ai_answers_field)
            if isinstance(human_answers, list) and len(human_answers) == 0:
                empty_human_answers_rows += 1
            if isinstance(ai_answers, list) and len(ai_answers) == 0:
                empty_ai_answers_rows += 1

        return Hc3SelectorValidationSummary(
            selector=split,
            dataset_rows=dataset_rows,
            loaded_rows=selected_size,
            missing_required_rows=missing_required_rows,
            empty_human_answers_rows=empty_human_answers_rows,
            empty_ai_answers_rows=empty_ai_answers_rows,
        )

    def validate_selectors(self, sample_rows: int) -> Hc3LoadCheckReport:
        """Run selector row checks for all configured selectors."""

        summaries = [self.validate_selector_rows(split=split, sample_rows=sample_rows) for split in self.list_splits()]
        return Hc3LoadCheckReport(
            dataset_id=self.dataset_id,
            source_uri=self._config.source_uri,
            requested_rows_per_selector=sample_rows,
            summaries=tuple(summaries),
        )
