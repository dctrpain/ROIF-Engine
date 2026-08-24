from __future__ import annotations

import json

from roif.development.expression_gate import (
    ExpressionDecision,
    ExpressionEvidence,
    ExpressionKind,
)
from roif.development.llm_verbalizer import VerbalizationRequest
from roif.development.ollama_backend import OllamaBackend


def _request() -> VerbalizationRequest:
    evidence = ExpressionEvidence(
        sequence_index=None,
        timestamp=None,
        deviation_score=None,
        maximum_absolute_deviation=None,
        l2_deviation_norm=None,
        group_count=0,
        strongest_group_strength=None,
        represented_dimension=3,
        residual_variance_fraction=0.45759965414789505,
        persistent_residual_dimension_count=2,
        growth_supported=True,
    )

    decision = ExpressionDecision(
        kind=ExpressionKind.REPRESENTATIONAL_INSUFFICIENCY,
        evidence=evidence,
    )

    return VerbalizationRequest(
        decision=decision,
        canonical_text=(
            "Моего текущего внутреннего представления недостаточно "
            "для полного описания повторяющейся структуры моего опыта."
        ),
        language="ru",
    )


def test_backend_exposes_configured_model():
    backend = OllamaBackend(model="gemma3:4b")

    assert backend.model == "gemma3:4b"


def test_empty_canonical_text_returns_empty_without_request():
    backend = OllamaBackend()

    original = _request()

    request = VerbalizationRequest(
        decision=original.decision,
        canonical_text="",
        language="ru",
    )

    assert backend.generate(request) == ""


def test_payload_contains_only_expression_level_evidence(monkeypatch):
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"response": "Допустимая переформулировка."}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )

    backend = OllamaBackend()
    result = backend.generate(_request())

    assert result == "Допустимая переформулировка."

    body = json.loads(
        captured["request"].data.decode("utf-8")
    )

    prompt = body["prompt"]

    assert "representational_insufficiency" in prompt
    assert '"represented_dimension": 3' in prompt
    assert '"persistent_residual_dimension_count": 2' in prompt
    assert '"growth_supported": true' in prompt

    assert "WorldState" not in prompt
    assert "BodyGroundTruth" not in prompt
    assert "perturbation_node" not in prompt
    assert "world coordinates" not in prompt


def test_empty_ollama_response_falls_back_to_canonical(monkeypatch):
    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"response": ""}
            ).encode("utf-8")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: DummyResponse(),
    )

    request = _request()

    backend = OllamaBackend()

    assert backend.generate(request) == request.canonical_text
