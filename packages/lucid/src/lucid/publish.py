from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OutputSpec:
    name: str
    kind: str
    config: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            **self.config,
        }


class PublishNamespace:
    @staticmethod
    def video(
        *,
        name: str,
        width: int,
        height: int,
        fps: int,
        pixel_format: str = "rgb24",
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="video",
            config={
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
                "pixel_format": pixel_format,
            },
        )

    @staticmethod
    def audio(
        *,
        name: str,
        sample_rate_hz: int,
        channels: int,
        sample_format: str = "float32",
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="audio",
            config={
                "sample_rate_hz": int(sample_rate_hz),
                "channels": int(channels),
                "sample_format": sample_format,
            },
        )

    @staticmethod
    def json(
        *,
        name: str,
        schema: dict[str, Any] | None = None,
        max_bytes: int = 16 * 1024,
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="json",
            config={
                "schema": schema or {"type": "object"},
                "max_bytes": int(max_bytes),
            },
        )

    @staticmethod
    def bytes(
        *,
        name: str,
        content_type: str = "application/octet-stream",
        max_bytes: int = 16 * 1024,
    ) -> OutputSpec:
        return OutputSpec(
            name=name,
            kind="bytes",
            config={
                "content_type": content_type,
                "max_bytes": int(max_bytes),
            },
        )


publish = PublishNamespace()
