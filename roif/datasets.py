"""
Dataset abstractions for ROIF Engine v1.

The module provides immutable dataset records, deterministic splitting,
filtering, batching, schema validation, statistics, and JSON persistence
integration through ``roif.serialization``.

It is intentionally domain-neutral. A record may represent a patient case,
synthetic graph, experiment, simulation run, or another ROIF-compatible sample.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, TypeAlias
import math
import random

from .serialization import (
    SerializationError,
    load_json,
    save_json,
    to_serializable,
)


class DatasetError(ValueError):
    """Raised when dataset input or configuration is invalid."""


RecordId: TypeAlias = Hashable
RecordPredicate: TypeAlias = Callable[["DatasetRecord"], bool]
RecordTransform: TypeAlias = Callable[["DatasetRecord"], "DatasetRecord"]


def _normalize_text(
    value: Any,
    *,
    name: str,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise DatasetError(f"{name} must be a string{suffix}.")
    result = value.strip()
    if not result:
        raise DatasetError(f"{name} cannot be empty.")
    return result


def _normalize_id(value: Any, *, name: str = "record_id") -> RecordId:
    if value is None:
        raise DatasetError(f"{name} cannot be None.")
    try:
        hash(value)
    except TypeError as exc:
        raise DatasetError(f"{name} must be hashable.") from exc
    if isinstance(value, str):
        return _normalize_text(value, name=name)
    return value


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise DatasetError(f"{name} must be a mapping.")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        copied[_normalize_text(key, name=f"{name} key")] = item
    return MappingProxyType(copied)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatasetError(f"{name} must be an integer >= 1.")
    return value


def _ratio(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise DatasetError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{name} must be a real number.") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise DatasetError(f"{name} must be between 0 and 1.")
    return result


class DatasetKind(str, Enum):
    """Semantic kind of dataset."""

    CLINICAL = "clinical"
    SYNTHETIC = "synthetic"
    EXPERIMENTAL = "experimental"
    BENCHMARK = "benchmark"
    MIXED = "mixed"
    CUSTOM = "custom"


class SplitName(str, Enum):
    """Canonical split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One immutable dataset sample."""

    record_id: RecordId
    features: Mapping[str, Any]
    target: Any = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "record_id",
            _normalize_id(self.record_id),
        )
        object.__setattr__(
            self,
            "features",
            _freeze_mapping(self.features, name="features"),
        )
        object.__setattr__(
            self,
            "group",
            _normalize_text(
                self.group,
                name="group",
                optional=True,
            ),
        )

        if isinstance(self.tags, (str, bytes)):
            raise DatasetError("tags must be a sequence of strings.")

        normalized_tags: list[str] = []
        seen: set[str] = set()
        for tag in self.tags:
            normalized = _normalize_text(tag, name="tag")
            if normalized in seen:
                raise DatasetError("tags cannot contain duplicates.")
            seen.add(normalized)
            normalized_tags.append(normalized)

        object.__setattr__(self, "tags", tuple(normalized_tags))
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "features": to_serializable(self.features),
            "target": to_serializable(self.target),
            "group": self.group,
            "tags": list(self.tags),
            "metadata": to_serializable(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Simple field-level schema for dataset records."""

    required_features: tuple[str, ...] = ()
    optional_features: tuple[str, ...] = ()
    allow_extra_features: bool = True
    require_target: bool = False

    def __post_init__(self) -> None:
        for name in ("required_features", "optional_features"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)):
                raise DatasetError(f"{name} must be a sequence.")
            normalized = tuple(
                _normalize_text(item, name=f"{name} item")
                for item in values
            )
            if len(normalized) != len(set(normalized)):
                raise DatasetError(f"{name} cannot contain duplicates.")
            object.__setattr__(self, name, normalized)

        overlap = set(self.required_features) & set(self.optional_features)
        if overlap:
            raise DatasetError(
                "required_features and optional_features cannot overlap."
            )

        if not isinstance(self.allow_extra_features, bool):
            raise DatasetError("allow_extra_features must be a bool.")
        if not isinstance(self.require_target, bool):
            raise DatasetError("require_target must be a bool.")

    def validate(self, record: DatasetRecord) -> None:
        """Validate one record against the schema."""
        if not isinstance(record, DatasetRecord):
            raise DatasetError("record must be a DatasetRecord.")

        feature_names = set(record.features)
        missing = set(self.required_features) - feature_names
        if missing:
            raise DatasetError(
                "Record is missing required features: "
                + ", ".join(sorted(missing))
                + "."
            )

        if not self.allow_extra_features:
            allowed = (
                set(self.required_features)
                | set(self.optional_features)
            )
            extra = feature_names - allowed
            if extra:
                raise DatasetError(
                    "Record contains unexpected features: "
                    + ", ".join(sorted(extra))
                    + "."
                )

        if self.require_target and record.target is None:
            raise DatasetError("Record target is required.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_features": list(self.required_features),
            "optional_features": list(self.optional_features),
            "allow_extra_features": self.allow_extra_features,
            "require_target": self.require_target,
        }


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Aggregate dataset statistics."""

    record_count: int
    feature_counts: Mapping[str, int]
    target_count: int
    group_counts: Mapping[str, int]
    tag_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count < 0
        ):
            raise DatasetError("record_count must be a non-negative integer.")

        if (
            isinstance(self.target_count, bool)
            or not isinstance(self.target_count, int)
            or self.target_count < 0
            or self.target_count > self.record_count
        ):
            raise DatasetError(
                "target_count must be between 0 and record_count."
            )

        for name in ("feature_counts", "group_counts", "tag_counts"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping):
                raise DatasetError(f"{name} must be a mapping.")
            parsed: dict[str, int] = {}
            for key, value in mapping.items():
                normalized = _normalize_text(key, name=f"{name} key")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise DatasetError(
                        f"{name} values must be non-negative integers."
                    )
                parsed[normalized] = value
            object.__setattr__(
                self,
                name,
                MappingProxyType(parsed),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "feature_counts": dict(self.feature_counts),
            "target_count": self.target_count,
            "group_counts": dict(self.group_counts),
            "tag_counts": dict(self.tag_counts),
        }


@dataclass(frozen=True, slots=True)
class Dataset:
    """Immutable ordered collection of unique dataset records."""

    name: str
    records: tuple[DatasetRecord, ...]
    kind: DatasetKind = DatasetKind.CUSTOM
    version: str = "1.0"
    description: str | None = None
    schema: DatasetSchema | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_text(self.name, name="name"),
        )

        records = tuple(self.records)
        if any(not isinstance(item, DatasetRecord) for item in records):
            raise DatasetError(
                "records must contain DatasetRecord values."
            )

        ids = [item.record_id for item in records]
        if len(ids) != len(set(ids)):
            raise DatasetError("record_id values must be unique.")

        if not isinstance(self.kind, DatasetKind):
            raise DatasetError("kind must be a DatasetKind.")

        object.__setattr__(
            self,
            "version",
            _normalize_text(self.version, name="version"),
        )
        object.__setattr__(
            self,
            "description",
            _normalize_text(
                self.description,
                name="description",
                optional=True,
            ),
        )

        if self.schema is not None and not isinstance(
            self.schema,
            DatasetSchema,
        ):
            raise DatasetError(
                "schema must be a DatasetSchema or None."
            )

        if self.schema is not None:
            for record in records:
                self.schema.validate(record)

        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.records)

    def __getitem__(self, index: int | slice) -> DatasetRecord | tuple[DatasetRecord, ...]:
        return self.records[index]

    @property
    def ids(self) -> tuple[RecordId, ...]:
        return tuple(item.record_id for item in self.records)

    def get(self, record_id: RecordId) -> DatasetRecord:
        normalized = _normalize_id(record_id)
        for record in self.records:
            if record.record_id == normalized:
                return record
        raise KeyError(normalized)

    def contains(self, record_id: RecordId) -> bool:
        normalized = _normalize_id(record_id)
        return any(record.record_id == normalized for record in self.records)

    def filter(
        self,
        predicate: RecordPredicate,
        *,
        name: str | None = None,
    ) -> "Dataset":
        if not callable(predicate):
            raise DatasetError("predicate must be callable.")
        return Dataset(
            name=name or f"{self.name}-filtered",
            records=tuple(
                record
                for record in self.records
                if predicate(record)
            ),
            kind=self.kind,
            version=self.version,
            description=self.description,
            schema=self.schema,
            metadata=self.metadata,
        )

    def map(
        self,
        transform: RecordTransform,
        *,
        name: str | None = None,
    ) -> "Dataset":
        if not callable(transform):
            raise DatasetError("transform must be callable.")
        transformed = tuple(transform(record) for record in self.records)
        if any(not isinstance(item, DatasetRecord) for item in transformed):
            raise DatasetError(
                "transform must return DatasetRecord values."
            )
        return Dataset(
            name=name or f"{self.name}-mapped",
            records=transformed,
            kind=self.kind,
            version=self.version,
            description=self.description,
            schema=self.schema,
            metadata=self.metadata,
        )

    def select(self, record_ids: Sequence[RecordId], *, name: str | None = None) -> "Dataset":
        if isinstance(record_ids, (str, bytes)):
            raise DatasetError(
                "record_ids must be a sequence of identifiers."
            )
        ids = tuple(_normalize_id(item) for item in record_ids)
        if len(ids) != len(set(ids)):
            raise DatasetError("record_ids cannot contain duplicates.")
        selected = tuple(self.get(item) for item in ids)
        return Dataset(
            name=name or f"{self.name}-selection",
            records=selected,
            kind=self.kind,
            version=self.version,
            description=self.description,
            schema=self.schema,
            metadata=self.metadata,
        )

    def shuffle(
        self,
        *,
        seed: int = 0,
        name: str | None = None,
    ) -> "Dataset":
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DatasetError("seed must be an integer.")
        records = list(self.records)
        random.Random(seed).shuffle(records)
        return Dataset(
            name=name or f"{self.name}-shuffled",
            records=tuple(records),
            kind=self.kind,
            version=self.version,
            description=self.description,
            schema=self.schema,
            metadata=self.metadata,
        )

    def batches(
        self,
        batch_size: int,
        *,
        drop_last: bool = False,
    ) -> tuple[tuple[DatasetRecord, ...], ...]:
        size = _positive_int(batch_size, name="batch_size")
        if not isinstance(drop_last, bool):
            raise DatasetError("drop_last must be a bool.")

        batches: list[tuple[DatasetRecord, ...]] = []
        for start in range(0, len(self.records), size):
            batch = self.records[start:start + size]
            if drop_last and len(batch) < size:
                continue
            batches.append(batch)
        return tuple(batches)

    def summary(self) -> DatasetSummary:
        feature_counts: Counter[str] = Counter()
        group_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        target_count = 0

        for record in self.records:
            feature_counts.update(record.features.keys())
            if record.target is not None:
                target_count += 1
            if record.group is not None:
                group_counts[record.group] += 1
            tag_counts.update(record.tags)

        return DatasetSummary(
            record_count=len(self.records),
            feature_counts=feature_counts,
            target_count=target_count,
            group_counts=group_counts,
            tag_counts=tag_counts,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "records": [item.as_dict() for item in self.records],
            "kind": self.kind.value,
            "version": self.version,
            "description": self.description,
            "schema": (
                None if self.schema is None else self.schema.as_dict()
            ),
            "metadata": to_serializable(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Train/validation/test split."""

    train: Dataset
    validation: Dataset
    test: Dataset
    seed: int
    stratified: bool

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            if not isinstance(getattr(self, name), Dataset):
                raise DatasetError(f"{name} must be a Dataset.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DatasetError("seed must be an integer.")
        if not isinstance(self.stratified, bool):
            raise DatasetError("stratified must be a bool.")

        all_ids = self.train.ids + self.validation.ids + self.test.ids
        if len(all_ids) != len(set(all_ids)):
            raise DatasetError("Dataset split contains overlapping records.")

    @property
    def total_count(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    def as_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "test": self.test.as_dict(),
            "seed": self.seed,
            "stratified": self.stratified,
            "total_count": self.total_count,
        }


def _split_counts(
    total: int,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if total < 0:
        raise DatasetError("total cannot be negative.")

    ratios = (
        _ratio(train_ratio, name="train_ratio"),
        _ratio(validation_ratio, name="validation_ratio"),
        _ratio(test_ratio, name="test_ratio"),
    )

    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise DatasetError("Split ratios must sum to 1.")

    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)

    remainders = sorted(
        range(3),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in remainders[:remaining]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def split_dataset(
    dataset: Dataset,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
    stratify_by: Callable[[DatasetRecord], Hashable] | None = None,
) -> DatasetSplit:
    """Deterministically split a dataset."""

    if not isinstance(dataset, Dataset):
        raise DatasetError("dataset must be a Dataset.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetError("seed must be an integer.")
    if stratify_by is not None and not callable(stratify_by):
        raise DatasetError("stratify_by must be callable or None.")

    train_count, validation_count, test_count = _split_counts(
        len(dataset),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    rng = random.Random(seed)

    if stratify_by is None:
        records = list(dataset.records)
        rng.shuffle(records)
        train_records = tuple(records[:train_count])
        validation_records = tuple(
            records[train_count:train_count + validation_count]
        )
        test_records = tuple(
            records[train_count + validation_count:]
        )
    else:
        groups: dict[Hashable, list[DatasetRecord]] = {}
        for record in dataset.records:
            key = stratify_by(record)
            try:
                hash(key)
            except TypeError as exc:
                raise DatasetError(
                    "stratify_by must return hashable values."
                ) from exc
            groups.setdefault(key, []).append(record)

        train_records_list: list[DatasetRecord] = []
        validation_records_list: list[DatasetRecord] = []
        test_records_list: list[DatasetRecord] = []

        for key in sorted(groups, key=repr):
            group_records = groups[key]
            rng.shuffle(group_records)
            group_counts = _split_counts(
                len(group_records),
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
            )
            t_count, v_count, _ = group_counts
            train_records_list.extend(group_records[:t_count])
            validation_records_list.extend(
                group_records[t_count:t_count + v_count]
            )
            test_records_list.extend(
                group_records[t_count + v_count:]
            )

        rng.shuffle(train_records_list)
        rng.shuffle(validation_records_list)
        rng.shuffle(test_records_list)
        train_records = tuple(train_records_list)
        validation_records = tuple(validation_records_list)
        test_records = tuple(test_records_list)

    def build(name: SplitName, records: tuple[DatasetRecord, ...]) -> Dataset:
        return Dataset(
            name=f"{dataset.name}-{name.value}",
            records=records,
            kind=dataset.kind,
            version=dataset.version,
            description=dataset.description,
            schema=dataset.schema,
            metadata={
                **dict(dataset.metadata),
                "split": name.value,
                "source_dataset": dataset.name,
            },
        )

    return DatasetSplit(
        train=build(SplitName.TRAIN, train_records),
        validation=build(SplitName.VALIDATION, validation_records),
        test=build(SplitName.TEST, test_records),
        seed=seed,
        stratified=stratify_by is not None,
    )


def dataset_from_mapping(value: Mapping[str, Any]) -> Dataset:
    """Reconstruct a Dataset from its serialized mapping."""

    if not isinstance(value, Mapping):
        raise DatasetError("value must be a mapping.")

    try:
        kind = DatasetKind(value["kind"])
        schema_payload = value.get("schema")
        schema = (
            None
            if schema_payload is None
            else DatasetSchema(
                required_features=tuple(
                    schema_payload.get("required_features", ())
                ),
                optional_features=tuple(
                    schema_payload.get("optional_features", ())
                ),
                allow_extra_features=schema_payload.get(
                    "allow_extra_features", True
                ),
                require_target=schema_payload.get(
                    "require_target", False
                ),
            )
        )

        records = tuple(
            DatasetRecord(
                record_id=item["record_id"],
                features=item["features"],
                target=item.get("target"),
                group=item.get("group"),
                tags=tuple(item.get("tags", ())),
                metadata=item.get("metadata", {}),
            )
            for item in value["records"]
        )

        return Dataset(
            name=value["name"],
            records=records,
            kind=kind,
            version=value.get("version", "1.0"),
            description=value.get("description"),
            schema=schema,
            metadata=value.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetError("Invalid serialized dataset mapping.") from exc


def save_dataset(
    dataset: Dataset,
    path: str | Path,
) -> Path:
    """Save a Dataset as JSON."""
    if not isinstance(dataset, Dataset):
        raise DatasetError("dataset must be a Dataset.")
    try:
        return save_json(dataset, path)
    except SerializationError as exc:
        raise DatasetError(str(exc)) from exc


def load_dataset(path: str | Path) -> Dataset:
    """Load a Dataset from JSON."""
    try:
        value = load_json(path)
    except SerializationError as exc:
        raise DatasetError(str(exc)) from exc
    if not isinstance(value, Mapping):
        raise DatasetError("Dataset JSON must contain an object.")
    return dataset_from_mapping(value)


__all__ = [
    "Dataset",
    "DatasetError",
    "DatasetKind",
    "DatasetRecord",
    "DatasetSchema",
    "DatasetSplit",
    "DatasetSummary",
    "RecordId",
    "RecordPredicate",
    "RecordTransform",
    "SplitName",
    "dataset_from_mapping",
    "load_dataset",
    "save_dataset",
    "split_dataset",
]
