from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BardMode:
    name: str
    description: str


@dataclass(frozen=True)
class BardConfig:
    experiment_stage: str = "early experiment"
    product_name: str = "Bard"
    principles_reference: str = "AI Principles"


def format_bard_response(*, mode: str, content: str) -> str:
    return f"[{str(mode).strip()}] {str(content).strip()}"


class BardSession:
    def __init__(self, config: BardConfig) -> None:
        self.config = config
        self._modes = (
            BardMode("productivity", "help users maximize their time"),
            BardMode("creativity", "spark imagination and drafting"),
            BardMode("curiosity", "support exploration and learning"),
        )

    def modes(self) -> tuple[str, ...]:
        return tuple(mode.name for mode in self._modes)

    def safety_note(self) -> str:
        return f"Responses are guided by {self.config.principles_reference} and ongoing user feedback."


__all__ = [
    "BardConfig",
    "BardMode",
    "BardSession",
    "format_bard_response",
]
