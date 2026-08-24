from __future__ import annotations

import json
import urllib.error
import urllib.request

from .llm_verbalizer import VerbalizationRequest


class OllamaBackend:
    """
    Local Ollama backend used only as a linguistic verbalizer.

    Fundamental invariant:

        Ollama does not determine RDA internal state.

    It receives only content already admitted by ExpressionGate.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        request: VerbalizationRequest,
    ) -> str:
        if not request.canonical_text.strip():
            return ""

        evidence = request.decision.evidence

        allowed_evidence = {
            "kind": request.decision.kind.value,
            "deviation_score": evidence.deviation_score,
            "group_count": evidence.group_count,
            "strongest_group_strength": (
                evidence.strongest_group_strength
            ),
            "represented_dimension": (
                evidence.represented_dimension
            ),
            "residual_variance_fraction": (
                evidence.residual_variance_fraction
            ),
            "persistent_residual_dimension_count": (
                evidence.persistent_residual_dimension_count
            ),
            "growth_supported": evidence.growth_supported,
        }

        prompt = (
            "Ты являешься только языковым адаптером.\n"
            "Ты НЕ интерпретируешь внутреннее состояние системы.\n"
            "Ты НЕ добавляешь причины, эмоции, намерения, цели, "
            "объекты мира, ощущения, оценки или новые понятия.\n"
            "Ты НЕ заменяешь технические понятия более сильными "
            "утверждениями.\n\n"
            "Разрешено только сделать исходное высказывание "
            "естественнее на русском языке, сохранив его содержание.\n\n"
            "Если безопасно переформулировать невозможно, "
            "верни исходный текст дословно.\n\n"
            f"Разрешённые данные:\n"
            f"{json.dumps(allowed_evidence, ensure_ascii=False)}\n\n"
            f"Исходное высказывание:\n"
            f"{request.canonical_text}\n\n"
            "Верни только итоговую фразу без пояснений."
        )

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        http_request = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self._timeout,
            ) as response:
                raw = response.read().decode("utf-8")

        except (
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}"
            ) from exc

        parsed = json.loads(raw)

        text = str(
            parsed.get("response", "")
        ).strip()

        if not text:
            return request.canonical_text

        return text
