"""Tests for roif.serialization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any
import json

import numpy as np
import pytest

from roif.serialization import (
    ChecksumAlgorithm,
    MigrationRegistry,
    MigrationStep,
    NPZBundle,
    SerializationEnvelope,
    SerializationError,
    SerializationFormat,
    canonical_json_bytes,
    compute_checksum,
    decode_envelope,
    dumps_csv,
    dumps_json,
    dumps_yaml,
    envelope_from_mapping,
    infer_format,
    load_csv,
    load_json,
    load_json_envelope,
    load_npz,
    load_yaml,
    loads_csv,
    loads_json,
    loads_yaml,
    make_envelope,
    require_valid_envelope,
    save_csv,
    save_json,
    save_npz,
    save_yaml,
    to_serializable,
    utc_now_iso,
    verify_envelope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ExampleEnum(str, Enum):
    VALUE = "value"


@dataclass
class ExampleDataclass:
    number: int
    label: str


class ExampleResult:
    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": 42,
            "valid": True,
        }


def make_envelope_fixture(
    *,
    payload: Any = None,
    schema_version: str = "1.0",
) -> SerializationEnvelope:
    if payload is None:
        payload = {"answer": 42}

    return make_envelope(
        payload,
        object_type="ExampleResult",
        schema_version=schema_version,
        producer="ROIF Test",
        producer_version="1.0",
        created_at="2026-08-01T09:00:00Z",
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


def test_serialization_format_values_are_stable() -> None:
    assert tuple(item.value for item in SerializationFormat) == (
        "json",
        "yaml",
        "csv",
        "npz",
    )


def test_checksum_algorithm_values_are_stable() -> None:
    assert tuple(item.value for item in ChecksumAlgorithm) == (
        "sha256",
        "sha512",
    )


# ---------------------------------------------------------------------------
# to_serializable
# ---------------------------------------------------------------------------


def test_to_serializable_accepts_json_scalars() -> None:
    assert to_serializable(None) is None
    assert to_serializable("text") == "text"
    assert to_serializable(3) == 3
    assert to_serializable(2.5) == pytest.approx(2.5)
    assert to_serializable(True) is True


def test_to_serializable_accepts_nested_mappings_and_sequences() -> None:
    result = to_serializable(
        {
            "tuple": (1, 2),
            "list": [3, 4],
            "nested": {"ok": True},
        }
    )

    assert result == {
        "tuple": [1, 2],
        "list": [3, 4],
        "nested": {"ok": True},
    }


def test_to_serializable_accepts_enum() -> None:
    assert to_serializable(ExampleEnum.VALUE) == "value"


def test_to_serializable_accepts_dataclass() -> None:
    result = to_serializable(
        ExampleDataclass(number=7, label="test")
    )

    assert result == {
        "number": 7,
        "label": "test",
    }


def test_to_serializable_accepts_as_dict_object() -> None:
    assert to_serializable(ExampleResult()) == {
        "answer": 42,
        "valid": True,
    }


def test_to_serializable_accepts_numpy_scalar() -> None:
    assert to_serializable(np.float64(1.25)) == pytest.approx(1.25)


def test_to_serializable_accepts_numpy_array() -> None:
    result = to_serializable(
        np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    )

    assert result == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]


def test_to_serializable_sorts_sets_deterministically() -> None:
    first = to_serializable({"B", "A", "C"})
    second = to_serializable({"C", "B", "A"})

    assert first == ["A", "B", "C"]
    assert first == second


def test_to_serializable_can_reject_sets() -> None:
    with pytest.raises(SerializationError):
        to_serializable({"A", "B"}, allow_sets=False)


@pytest.mark.parametrize(
    "value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_to_serializable_rejects_nonfinite_float(
    value: float,
) -> None:
    with pytest.raises(SerializationError):
        to_serializable(value)


def test_to_serializable_rejects_nonfinite_numpy_array() -> None:
    with pytest.raises(SerializationError):
        to_serializable(
            np.asarray([1.0, float("nan")])
        )


@pytest.mark.parametrize(
    "value",
    (
        object(),
        lambda: None,
    ),
)
def test_to_serializable_rejects_unknown_type(value: Any) -> None:
    with pytest.raises(SerializationError):
        to_serializable(value)


@pytest.mark.parametrize(
    "mapping",
    (
        {1: "value"},
        {"": "value"},
        {"   ": "value"},
    ),
)
def test_to_serializable_rejects_invalid_mapping_keys(
    mapping: dict[Any, Any],
) -> None:
    with pytest.raises(SerializationError):
        to_serializable(mapping)


# ---------------------------------------------------------------------------
# Canonical JSON and checksums
# ---------------------------------------------------------------------------


def test_canonical_json_bytes_are_deterministic() -> None:
    first = canonical_json_bytes(
        {"b": 2, "a": 1}
    )
    second = canonical_json_bytes(
        {"a": 1, "b": 2}
    )

    assert first == second
    assert first == b'{"a":1,"b":2}'


def test_compute_checksum_sha256() -> None:
    checksum = compute_checksum(
        {"a": 1},
        algorithm=ChecksumAlgorithm.SHA256,
    )

    assert len(checksum) == 64
    assert checksum == compute_checksum({"a": 1})


def test_compute_checksum_sha512() -> None:
    checksum = compute_checksum(
        {"a": 1},
        algorithm=ChecksumAlgorithm.SHA512,
    )

    assert len(checksum) == 128


@pytest.mark.parametrize(
    "algorithm",
    (
        "sha256",
        None,
        1,
    ),
)
def test_compute_checksum_rejects_invalid_algorithm(
    algorithm: Any,
) -> None:
    with pytest.raises(SerializationError):
        compute_checksum(
            {"a": 1},
            algorithm=algorithm,
        )


# ---------------------------------------------------------------------------
# SerializationEnvelope
# ---------------------------------------------------------------------------


def test_make_envelope_properties() -> None:
    envelope = make_envelope_fixture()

    assert envelope.object_type == "ExampleResult"
    assert envelope.schema_version == "1.0"
    assert envelope.created_at == "2026-08-01T09:00:00Z"
    assert envelope.payload == {"answer": 42}
    assert envelope.checksum_algorithm is ChecksumAlgorithm.SHA256
    assert isinstance(envelope.metadata, MappingProxyType)
    assert verify_envelope(envelope) is True


def test_envelope_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        make_envelope_fixture().schema_version = "2.0"  # type: ignore[misc]


def test_envelope_as_dict() -> None:
    payload = make_envelope_fixture().as_dict()

    assert payload["object_type"] == "ExampleResult"
    assert payload["checksum_algorithm"] == "sha256"
    assert payload["metadata"] == {"source": "test"}


def test_envelope_normalizes_checksum_case() -> None:
    original = make_envelope_fixture()

    envelope = SerializationEnvelope(
        object_type=original.object_type,
        schema_version=original.schema_version,
        created_at=original.created_at,
        payload=original.payload,
        checksum=original.checksum.upper(),
        checksum_algorithm=original.checksum_algorithm,
        producer=original.producer,
        producer_version=original.producer_version,
    )

    assert envelope.checksum == original.checksum


def test_envelope_rejects_nonhex_checksum() -> None:
    with pytest.raises(SerializationError):
        SerializationEnvelope(
            object_type="Example",
            schema_version="1.0",
            created_at="2026-08-01T09:00:00Z",
            payload={"a": 1},
            checksum="not-hex",
        )


@pytest.mark.parametrize(
    "created_at",
    (
        "",
        "not-a-date",
        1,
    ),
)
def test_envelope_rejects_invalid_timestamp(
    created_at: Any,
) -> None:
    with pytest.raises(SerializationError):
        SerializationEnvelope(
            object_type="Example",
            schema_version="1.0",
            created_at=created_at,
            payload={"a": 1},
            checksum="00",
        )


def test_verify_envelope_detects_tampering() -> None:
    original = make_envelope_fixture()

    tampered = SerializationEnvelope(
        object_type=original.object_type,
        schema_version=original.schema_version,
        created_at=original.created_at,
        payload={"answer": 99},
        checksum=original.checksum,
        checksum_algorithm=original.checksum_algorithm,
        producer=original.producer,
        producer_version=original.producer_version,
    )

    assert verify_envelope(tampered) is False

    with pytest.raises(
        SerializationError,
        match="checksum verification failed",
    ):
        require_valid_envelope(tampered)


def test_envelope_from_mapping_round_trip() -> None:
    original = make_envelope_fixture()

    restored = envelope_from_mapping(
        original.as_dict()
    )

    assert restored == original
    assert verify_envelope(restored)


def test_envelope_from_mapping_rejects_missing_field() -> None:
    payload = make_envelope_fixture().as_dict()
    del payload["checksum"]

    with pytest.raises(
        SerializationError,
        match="missing required fields",
    ):
        envelope_from_mapping(payload)


def test_envelope_from_mapping_rejects_unknown_field() -> None:
    payload = make_envelope_fixture().as_dict()
    payload["unknown"] = 1

    with pytest.raises(
        SerializationError,
        match="unknown fields",
    ):
        envelope_from_mapping(payload)


def test_decode_envelope() -> None:
    envelope = make_envelope_fixture(
        payload={"value": 5}
    )

    result = decode_envelope(
        envelope,
        lambda payload: payload["value"] * 2,
    )

    assert result == 10


def test_decode_envelope_requires_mapping_payload() -> None:
    envelope = make_envelope_fixture(
        payload=[1, 2, 3]
    )

    with pytest.raises(
        SerializationError,
        match="mapping payload",
    ):
        decode_envelope(
            envelope,
            lambda payload: payload,
        )


@pytest.mark.parametrize(
    "decoder",
    (
        None,
        "bad",
        1,
    ),
)
def test_decode_envelope_rejects_invalid_decoder(
    decoder: Any,
) -> None:
    with pytest.raises(SerializationError):
        decode_envelope(
            make_envelope_fixture(),
            decoder,
        )


# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------


def test_migration_step_properties() -> None:
    step = MigrationStep(
        object_type="Example",
        from_version="1.0",
        to_version="2.0",
        migrate=lambda payload: {
            **payload,
            "version": 2,
        },
    )

    assert step.object_type == "Example"
    assert step.from_version == "1.0"
    assert callable(step.migrate)


def test_migration_step_rejects_equal_versions() -> None:
    with pytest.raises(SerializationError):
        MigrationStep(
            "Example",
            "1.0",
            "1.0",
            lambda payload: payload,
        )


def test_migration_step_requires_callable() -> None:
    with pytest.raises(SerializationError):
        MigrationStep(
            "Example",
            "1.0",
            "2.0",
            None,  # type: ignore[arg-type]
        )


def test_migration_registry_migrates_multiple_steps() -> None:
    registry = MigrationRegistry()

    registry.register(
        MigrationStep(
            "Example",
            "1.0",
            "2.0",
            lambda payload: {
                **payload,
                "b": payload["a"] + 1,
            },
        )
    )
    registry.register(
        MigrationStep(
            "Example",
            "2.0",
            "3.0",
            lambda payload: {
                **payload,
                "c": payload["b"] + 1,
            },
        )
    )

    result = registry.migrate_payload(
        object_type="Example",
        payload={"a": 1},
        from_version="1.0",
        to_version="3.0",
    )

    assert result == {
        "a": 1,
        "b": 2,
        "c": 3,
    }
    assert isinstance(result, MappingProxyType)


def test_migration_registry_rejects_duplicate_source() -> None:
    registry = MigrationRegistry()
    step = MigrationStep(
        "Example",
        "1.0",
        "2.0",
        lambda payload: payload,
    )
    registry.register(step)

    with pytest.raises(
        SerializationError,
        match="already registered",
    ):
        registry.register(step)


def test_migration_registry_detects_missing_path() -> None:
    with pytest.raises(
        SerializationError,
        match="No migration path",
    ):
        MigrationRegistry().migrate_payload(
            object_type="Example",
            payload={"a": 1},
            from_version="1.0",
            to_version="2.0",
        )


def test_migration_registry_detects_cycle() -> None:
    registry = MigrationRegistry()
    registry.register(
        MigrationStep(
            "Example",
            "1.0",
            "2.0",
            lambda payload: payload,
        )
    )
    registry.register(
        MigrationStep(
            "Example",
            "2.0",
            "1.0",
            lambda payload: payload,
        )
    )

    with pytest.raises(
        SerializationError,
        match="cycle detected",
    ):
        registry.migrate_payload(
            object_type="Example",
            payload={"a": 1},
            from_version="1.0",
            to_version="3.0",
        )


def test_migration_registry_rejects_nonmapping_result() -> None:
    registry = MigrationRegistry().register(
        MigrationStep(
            "Example",
            "1.0",
            "2.0",
            lambda payload: [1, 2],  # type: ignore[return-value]
        )
    )

    with pytest.raises(
        SerializationError,
        match="must return mappings",
    ):
        registry.migrate_payload(
            object_type="Example",
            payload={"a": 1},
            from_version="1.0",
            to_version="2.0",
        )


@pytest.mark.parametrize(
    "maximum_steps",
    (
        0,
        -1,
        True,
        1.5,
        "2",
    ),
)
def test_migration_registry_rejects_invalid_maximum_steps(
    maximum_steps: Any,
) -> None:
    with pytest.raises(SerializationError):
        MigrationRegistry().migrate_payload(
            object_type="Example",
            payload={"a": 1},
            from_version="1.0",
            to_version="1.0",
            maximum_steps=maximum_steps,
        )


def test_migration_registry_migrates_envelope() -> None:
    registry = MigrationRegistry().register(
        MigrationStep(
            "ExampleResult",
            "1.0",
            "2.0",
            lambda payload: {
                **payload,
                "new_field": True,
            },
        )
    )

    original = make_envelope_fixture()
    migrated = registry.migrate_envelope(
        original,
        to_version="2.0",
    )

    assert migrated.schema_version == "2.0"
    assert migrated.payload["new_field"] is True
    assert migrated.metadata["migrated_from"] == "1.0"
    assert verify_envelope(migrated)


def test_migration_registry_rejects_sequence_payload_envelope() -> None:
    registry = MigrationRegistry()
    envelope = make_envelope_fixture(
        payload=[1, 2]
    )

    with pytest.raises(
        SerializationError,
        match="mapping payloads",
    ):
        registry.migrate_envelope(
            envelope,
            to_version="2.0",
        )


# ---------------------------------------------------------------------------
# JSON text and files
# ---------------------------------------------------------------------------


def test_dumps_and_loads_json_round_trip() -> None:
    text = dumps_json(
        {
            "text": "Привет",
            "values": (1, 2),
        },
        sort_keys=True,
    )

    restored = loads_json(text)

    assert restored == {
        "text": "Привет",
        "values": [1, 2],
    }


def test_dumps_json_compact_mode() -> None:
    text = dumps_json(
        {"a": 1},
        indent=None,
    )

    assert "\n" not in text


@pytest.mark.parametrize(
    "indent",
    (
        -1,
        True,
        1.5,
        "2",
    ),
)
def test_dumps_json_rejects_invalid_indent(indent: Any) -> None:
    with pytest.raises(SerializationError):
        dumps_json(
            {"a": 1},
            indent=indent,
        )


@pytest.mark.parametrize(
    "text",
    (
        "",
        "   ",
        "{bad json}",
        None,
        1,
    ),
)
def test_loads_json_rejects_invalid_text(text: Any) -> None:
    with pytest.raises(SerializationError):
        loads_json(text)


def test_save_and_load_json_file(tmp_path) -> None:
    path = tmp_path / "nested" / "result.json"

    saved = save_json(
        {"a": 1, "text": "ROIF"},
        path,
    )
    loaded = load_json(path)

    assert saved == path
    assert loaded == {
        "a": 1,
        "text": "ROIF",
    }


def test_save_json_nonatomic(tmp_path) -> None:
    path = tmp_path / "result.json"

    save_json(
        {"a": 1},
        path,
        atomic=False,
    )

    assert json.loads(
        path.read_text(encoding="utf-8")
    ) == {"a": 1}


def test_load_json_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(
        SerializationError,
        match="does not exist",
    ):
        load_json(
            tmp_path / "missing.json"
        )


def test_save_and_load_json_envelope(tmp_path) -> None:
    path = tmp_path / "envelope.json"
    original = make_envelope_fixture()

    save_json(original, path)
    restored = load_json_envelope(path)

    assert restored == original
    assert verify_envelope(restored)


def test_load_json_envelope_detects_tampering(tmp_path) -> None:
    path = tmp_path / "envelope.json"
    payload = make_envelope_fixture().as_dict()
    payload["payload"]["answer"] = 99
    save_json(payload, path)

    with pytest.raises(
        SerializationError,
        match="checksum verification failed",
    ):
        load_json_envelope(path)


def test_load_json_envelope_can_skip_verification(tmp_path) -> None:
    path = tmp_path / "envelope.json"
    payload = make_envelope_fixture().as_dict()
    payload["payload"]["answer"] = 99
    save_json(payload, path)

    restored = load_json_envelope(
        path,
        verify=False,
    )

    assert restored.payload["answer"] == 99
    assert verify_envelope(restored) is False


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def test_yaml_round_trip_when_available() -> None:
    pytest.importorskip("yaml")

    text = dumps_yaml(
        {
            "text": "Привет",
            "values": [1, 2],
        }
    )
    restored = loads_yaml(text)

    assert restored == {
        "text": "Привет",
        "values": [1, 2],
    }


def test_yaml_file_round_trip_when_available(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "result.yaml"

    save_yaml(
        {"a": 1},
        path,
    )

    assert load_yaml(path) == {"a": 1}


def test_load_yaml_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(
        SerializationError,
        match="does not exist",
    ):
        load_yaml(
            tmp_path / "missing.yaml"
        )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_dumps_and_loads_csv_round_trip() -> None:
    text = dumps_csv(
        (
            {
                "id": 1,
                "name": "A",
                "metadata": {"x": 1},
            },
            {
                "id": 2,
                "name": "B",
                "metadata": [1, 2],
            },
        )
    )

    restored = loads_csv(text)

    assert restored[0]["id"] == "1"
    assert restored[0]["name"] == "A"
    assert restored[0]["metadata"] == '{"x":1}'
    assert restored[1]["metadata"] == "[1,2]"
    assert isinstance(restored[0], MappingProxyType)


def test_dumps_csv_respects_explicit_field_order() -> None:
    text = dumps_csv(
        ({"a": 1, "b": 2},),
        fieldnames=("b", "a"),
    )

    assert text.splitlines()[0] == "b,a"


def test_dumps_csv_supports_custom_delimiter() -> None:
    text = dumps_csv(
        ({"a": 1, "b": 2},),
        delimiter=";",
    )

    assert text.splitlines()[0] == "a;b"


@pytest.mark.parametrize(
    "rows",
    (
        "bad",
        (1,),
    ),
)
def test_dumps_csv_rejects_invalid_rows(rows: Any) -> None:
    with pytest.raises(SerializationError):
        dumps_csv(rows)


@pytest.mark.parametrize(
    "delimiter",
    (
        "",
        "::",
        None,
        1,
    ),
)
def test_csv_rejects_invalid_delimiter(
    delimiter: Any,
) -> None:
    with pytest.raises(SerializationError):
        dumps_csv(
            ({"a": 1},),
            delimiter=delimiter,
        )


def test_dumps_csv_rejects_duplicate_fieldnames() -> None:
    with pytest.raises(SerializationError):
        dumps_csv(
            ({"a": 1},),
            fieldnames=("a", "a"),
        )


def test_loads_csv_rejects_extra_unnamed_values() -> None:
    with pytest.raises(
        SerializationError,
        match="extra unnamed values",
    ):
        loads_csv(
            "a,b\n1,2,3\n"
        )


def test_save_and_load_csv_file(tmp_path) -> None:
    path = tmp_path / "rows.csv"

    save_csv(
        (
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
        ),
        path,
    )
    restored = load_csv(path)

    assert restored == (
        {"id": "1", "name": "A"},
        {"id": "2", "name": "B"},
    )


def test_load_csv_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(
        SerializationError,
        match="does not exist",
    ):
        load_csv(
            tmp_path / "missing.csv"
        )


# ---------------------------------------------------------------------------
# NPZ
# ---------------------------------------------------------------------------


def test_npz_bundle_properties() -> None:
    bundle = NPZBundle(
        arrays={
            "state": np.asarray([1.0, 2.0]),
            "ids": np.asarray([1, 2]),
        },
        metadata={"experiment": "E1"},
    )

    assert isinstance(bundle.arrays, MappingProxyType)
    assert isinstance(bundle.metadata, MappingProxyType)
    assert bundle.arrays["state"].flags.writeable is False
    assert bundle.as_dict()["arrays"]["state"] == [1.0, 2.0]


def test_npz_bundle_copies_source_arrays() -> None:
    source = np.asarray([1.0, 2.0])
    bundle = NPZBundle(arrays={"state": source})
    source[0] = 99.0

    assert bundle.arrays["state"][0] == pytest.approx(1.0)


def test_npz_bundle_rejects_object_dtype() -> None:
    with pytest.raises(
        SerializationError,
        match="object dtype",
    ):
        NPZBundle(
            arrays={
                "objects": np.asarray(
                    [{"a": 1}],
                    dtype=object,
                )
            }
        )


def test_npz_bundle_rejects_nonfinite_values() -> None:
    with pytest.raises(
        SerializationError,
        match="finite values",
    ):
        NPZBundle(
            arrays={
                "state": np.asarray(
                    [1.0, float("nan")]
                )
            }
        )


def test_save_and_load_npz_round_trip(tmp_path) -> None:
    path = tmp_path / "bundle.npz"

    save_npz(
        {
            "state": np.asarray([1.0, 2.0]),
            "matrix": np.asarray(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        },
        path,
        metadata={"experiment": "E1"},
    )

    restored = load_npz(path)

    assert np.allclose(
        restored.arrays["state"],
        (1.0, 2.0),
    )
    assert np.array_equal(
        restored.arrays["matrix"],
        np.asarray(
            [
                [1, 2],
                [3, 4],
            ]
        ),
    )
    assert restored.metadata["experiment"] == "E1"


def test_save_npz_accepts_bundle(tmp_path) -> None:
    path = tmp_path / "bundle.npz"
    bundle = NPZBundle(
        arrays={"state": np.asarray([1.0])},
        metadata={"source": "bundle"},
    )

    save_npz(bundle, path)
    restored = load_npz(path)

    assert restored.metadata["source"] == "bundle"


def test_save_npz_rejects_metadata_with_bundle(tmp_path) -> None:
    bundle = NPZBundle(
        arrays={"state": np.asarray([1.0])}
    )

    with pytest.raises(SerializationError):
        save_npz(
            bundle,
            tmp_path / "bundle.npz",
            metadata={"bad": True},
        )


def test_load_npz_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(
        SerializationError,
        match="does not exist",
    ):
        load_npz(
            tmp_path / "missing.npz"
        )


# ---------------------------------------------------------------------------
# Format inference and time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    (
        ("result.json", SerializationFormat.JSON),
        ("result.yaml", SerializationFormat.YAML),
        ("result.yml", SerializationFormat.YAML),
        ("rows.csv", SerializationFormat.CSV),
        ("arrays.npz", SerializationFormat.NPZ),
        ("RESULT.JSON", SerializationFormat.JSON),
    ),
)
def test_infer_format(
    filename: str,
    expected: SerializationFormat,
) -> None:
    assert infer_format(filename) is expected


@pytest.mark.parametrize(
    "filename",
    (
        "result",
        "result.txt",
        "result.pkl",
    ),
)
def test_infer_format_rejects_unknown_extension(
    filename: str,
) -> None:
    with pytest.raises(SerializationError):
        infer_format(filename)


def test_utc_now_iso() -> None:
    value = utc_now_iso()

    assert value.endswith("Z")
    assert "T" in value
