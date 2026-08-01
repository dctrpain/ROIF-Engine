"""Tests for roif.datasets."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import Any

import pytest

from roif.datasets import (
    Dataset,
    DatasetError,
    DatasetKind,
    DatasetRecord,
    DatasetSchema,
    DatasetSplit,
    DatasetSummary,
    SplitName,
    dataset_from_mapping,
    load_dataset,
    save_dataset,
    split_dataset,
)


def record(
    record_id: Any = "r1",
    *,
    target: Any = 1,
    group: str | None = "A",
    tags: tuple[str, ...] = ("tag",),
    features: dict[str, Any] | None = None,
) -> DatasetRecord:
    return DatasetRecord(
        record_id=record_id,
        features=features or {"x": 1.0, "y": 2.0},
        target=target,
        group=group,
        tags=tags,
        metadata={"source": "test"},
    )


def schema() -> DatasetSchema:
    return DatasetSchema(
        required_features=("x",),
        optional_features=("y",),
        allow_extra_features=False,
        require_target=True,
    )


def dataset(
    *,
    count: int = 6,
    with_schema: bool = False,
) -> Dataset:
    records = tuple(
        DatasetRecord(
            record_id=f"r{index}",
            features={"x": float(index), "y": index + 1},
            target=index % 2,
            group="A" if index % 2 == 0 else "B",
            tags=("even",) if index % 2 == 0 else ("odd",),
        )
        for index in range(count)
    )
    return Dataset(
        name="demo",
        records=records,
        kind=DatasetKind.SYNTHETIC,
        version="1.0",
        description="Demo dataset.",
        schema=schema() if with_schema else None,
        metadata={"experiment": "E1"},
    )


def test_enum_values() -> None:
    assert tuple(item.value for item in DatasetKind) == (
        "clinical",
        "synthetic",
        "experimental",
        "benchmark",
        "mixed",
        "custom",
    )
    assert tuple(item.value for item in SplitName) == (
        "train",
        "validation",
        "test",
    )


def test_record_properties() -> None:
    item = record()
    assert item.record_id == "r1"
    assert item.features["x"] == pytest.approx(1.0)
    assert isinstance(item.features, MappingProxyType)
    assert isinstance(item.metadata, MappingProxyType)
    assert item.as_dict()["tags"] == ["tag"]


def test_record_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        record().record_id = "x"  # type: ignore[misc]


@pytest.mark.parametrize("value", (None, [], {}))
def test_record_rejects_invalid_id(value: Any) -> None:
    with pytest.raises(DatasetError):
        record(record_id=value)


def test_record_normalizes_string_id() -> None:
    assert record(record_id="  r1  ").record_id == "r1"


def test_record_rejects_duplicate_tags() -> None:
    with pytest.raises(DatasetError):
        record(tags=("a", "a"))


@pytest.mark.parametrize("tags", ("bad", (1,)))
def test_record_rejects_invalid_tags(tags: Any) -> None:
    with pytest.raises(DatasetError):
        record(tags=tags)


def test_schema_properties() -> None:
    item = schema()
    assert item.required_features == ("x",)
    assert item.optional_features == ("y",)
    assert item.require_target is True
    assert item.as_dict()["allow_extra_features"] is False


def test_schema_validates_record() -> None:
    schema().validate(record())


def test_schema_rejects_missing_feature() -> None:
    with pytest.raises(DatasetError, match="missing required"):
        schema().validate(
            record(features={"y": 2.0})
        )


def test_schema_rejects_extra_feature() -> None:
    with pytest.raises(DatasetError, match="unexpected features"):
        schema().validate(
            record(features={"x": 1.0, "y": 2.0, "z": 3.0})
        )


def test_schema_requires_target() -> None:
    with pytest.raises(DatasetError, match="target is required"):
        schema().validate(record(target=None))


def test_schema_rejects_overlap() -> None:
    with pytest.raises(DatasetError):
        DatasetSchema(
            required_features=("x",),
            optional_features=("x",),
        )


def test_dataset_properties() -> None:
    item = dataset()
    assert len(item) == 6
    assert item.ids[0] == "r0"
    assert item[0].record_id == "r0"
    assert tuple(record.record_id for record in item) == item.ids
    assert isinstance(item.metadata, MappingProxyType)


def test_dataset_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        dataset().name = "x"  # type: ignore[misc]


def test_dataset_rejects_duplicate_ids() -> None:
    with pytest.raises(DatasetError):
        Dataset(
            name="bad",
            records=(record("r1"), record("r1")),
        )


def test_dataset_validates_schema() -> None:
    item = dataset(with_schema=True)
    assert item.schema is not None


def test_dataset_schema_failure_propagates() -> None:
    with pytest.raises(DatasetError):
        Dataset(
            name="bad",
            records=(
                record(
                    features={"x": 1.0, "extra": 2.0}
                ),
            ),
            schema=schema(),
        )


def test_dataset_get_and_contains() -> None:
    item = dataset()
    assert item.get("r2").target == 0
    assert item.contains("r3")
    assert not item.contains("missing")
    with pytest.raises(KeyError):
        item.get("missing")


def test_dataset_filter() -> None:
    filtered = dataset().filter(
        lambda item: item.group == "A"
    )
    assert len(filtered) == 3
    assert all(item.group == "A" for item in filtered)


def test_dataset_filter_rejects_bad_predicate() -> None:
    with pytest.raises(DatasetError):
        dataset().filter(None)  # type: ignore[arg-type]


def test_dataset_map() -> None:
    mapped = dataset(count=2).map(
        lambda item: DatasetRecord(
            record_id=item.record_id,
            features={**dict(item.features), "z": 3},
            target=item.target,
            group=item.group,
            tags=item.tags,
        )
    )
    assert mapped[0].features["z"] == 3


def test_dataset_map_rejects_bad_result() -> None:
    with pytest.raises(DatasetError):
        dataset(count=1).map(
            lambda item: "bad"  # type: ignore[return-value]
        )


def test_dataset_select_preserves_requested_order() -> None:
    selected = dataset().select(("r3", "r1"))
    assert selected.ids == ("r3", "r1")


def test_dataset_select_rejects_duplicates() -> None:
    with pytest.raises(DatasetError):
        dataset().select(("r1", "r1"))


def test_dataset_shuffle_is_deterministic() -> None:
    first = dataset().shuffle(seed=42)
    second = dataset().shuffle(seed=42)
    assert first.ids == second.ids
    assert set(first.ids) == set(dataset().ids)


@pytest.mark.parametrize("seed", (True, 1.5, "42"))
def test_dataset_shuffle_rejects_bad_seed(seed: Any) -> None:
    with pytest.raises(DatasetError):
        dataset().shuffle(seed=seed)


def test_dataset_batches() -> None:
    batches = dataset(count=5).batches(2)
    assert tuple(len(batch) for batch in batches) == (2, 2, 1)


def test_dataset_batches_drop_last() -> None:
    batches = dataset(count=5).batches(2, drop_last=True)
    assert tuple(len(batch) for batch in batches) == (2, 2)


@pytest.mark.parametrize("size", (0, -1, True, 1.5, "2"))
def test_dataset_batches_rejects_bad_size(size: Any) -> None:
    with pytest.raises(DatasetError):
        dataset().batches(size)


def test_dataset_summary() -> None:
    summary = dataset().summary()
    assert summary.record_count == 6
    assert summary.target_count == 6
    assert summary.feature_counts == {"x": 6, "y": 6}
    assert summary.group_counts == {"A": 3, "B": 3}
    assert summary.tag_counts == {"even": 3, "odd": 3}


def test_dataset_summary_properties() -> None:
    summary = DatasetSummary(
        record_count=2,
        feature_counts={"x": 2},
        target_count=1,
        group_counts={"A": 2},
        tag_counts={"tag": 1},
    )
    assert isinstance(summary.feature_counts, MappingProxyType)
    assert summary.as_dict()["target_count"] == 1


def test_dataset_as_dict() -> None:
    payload = dataset(count=1, with_schema=True).as_dict()
    assert payload["kind"] == "synthetic"
    assert payload["records"][0]["record_id"] == "r0"
    assert payload["schema"]["required_features"] == ["x"]


def test_split_dataset_counts() -> None:
    result = split_dataset(
        dataset(count=10),
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )
    assert len(result.train) == 6
    assert len(result.validation) == 2
    assert len(result.test) == 2
    assert result.total_count == 10


def test_split_dataset_is_deterministic() -> None:
    first = split_dataset(dataset(count=10), seed=123)
    second = split_dataset(dataset(count=10), seed=123)
    assert first.train.ids == second.train.ids
    assert first.validation.ids == second.validation.ids
    assert first.test.ids == second.test.ids


def test_split_dataset_has_no_overlap() -> None:
    result = split_dataset(dataset(count=12), seed=5)
    all_ids = (
        result.train.ids
        + result.validation.ids
        + result.test.ids
    )
    assert len(all_ids) == len(set(all_ids))


def test_split_dataset_stratified() -> None:
    result = split_dataset(
        dataset(count=20),
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=1,
        stratify_by=lambda item: item.target,
    )
    assert result.stratified is True
    assert {item.target for item in result.train} == {0, 1}
    assert {item.target for item in result.validation} == {0, 1}
    assert {item.target for item in result.test} == {0, 1}


@pytest.mark.parametrize(
    "ratios",
    (
        (0.7, 0.2, 0.2),
        (-0.1, 0.5, 0.6),
        (1.1, 0.0, -0.1),
    ),
)
def test_split_dataset_rejects_bad_ratios(
    ratios: tuple[float, float, float],
) -> None:
    with pytest.raises(DatasetError):
        split_dataset(
            dataset(),
            train_ratio=ratios[0],
            validation_ratio=ratios[1],
            test_ratio=ratios[2],
        )


def test_split_dataset_rejects_bad_stratifier() -> None:
    with pytest.raises(DatasetError):
        split_dataset(
            dataset(),
            stratify_by="bad",  # type: ignore[arg-type]
        )


def test_split_object_validation() -> None:
    item = dataset(count=3)
    split = DatasetSplit(
        train=item.select(("r0",)),
        validation=item.select(("r1",)),
        test=item.select(("r2",)),
        seed=0,
        stratified=False,
    )
    assert split.total_count == 3


def test_split_rejects_overlap() -> None:
    item = dataset(count=2)
    with pytest.raises(DatasetError):
        DatasetSplit(
            train=item.select(("r0",)),
            validation=item.select(("r0",)),
            test=item.select(("r1",)),
            seed=0,
            stratified=False,
        )


def test_dataset_mapping_round_trip() -> None:
    original = dataset(count=3, with_schema=True)
    restored = dataset_from_mapping(original.as_dict())
    assert restored.as_dict() == original.as_dict()


def test_dataset_from_mapping_rejects_bad_input() -> None:
    with pytest.raises(DatasetError):
        dataset_from_mapping({"name": "missing"})


def test_save_and_load_dataset(tmp_path) -> None:
    path = tmp_path / "datasets" / "demo.json"
    original = dataset(count=4, with_schema=True)

    saved = save_dataset(original, path)
    restored = load_dataset(path)

    assert saved == path
    assert restored.as_dict() == original.as_dict()


def test_load_dataset_rejects_nonobject_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_dataset(path)
