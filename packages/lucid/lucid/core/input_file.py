from __future__ import annotations

import io
from dataclasses import dataclass
from types import UnionType
from typing import IO, Sequence, Union, get_args, get_origin

from pydantic import Field
from pydantic.fields import FieldInfo


_DEFAULT_IMAGE_MIME_TYPES = ("image/jpeg", "image/png", "image/webp")


@dataclass(frozen=True, slots=True)
class InputFile:
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    data: bytes

    def open(self) -> IO[bytes]:
        return io.BytesIO(self.data)


def file_input(
    *,
    mime_types: Sequence[str] | None = None,
    max_bytes: int = 8_000_000,
) -> FieldInfo:
    return Field(
        ...,
        json_schema_extra={
            "x-lucid-upload": {
                "kind": "file",
                "mime_types": list(_normalize_mime_types(mime_types)),
                "max_bytes": _normalize_max_bytes(max_bytes),
            }
        },
    )


def image_input(
    *,
    mime_types: Sequence[str] = _DEFAULT_IMAGE_MIME_TYPES,
    max_bytes: int = 1_000_000,
    size: tuple[int, int] | None = None,
) -> FieldInfo:
    upload = {
        "kind": "image",
        "mime_types": list(_normalize_mime_types(mime_types)),
        "max_bytes": _normalize_max_bytes(max_bytes),
    }
    if size is not None:
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("image_input() size must contain positive integers")
        upload["target_width"] = int(width)
        upload["target_height"] = int(height)
    return Field(..., json_schema_extra={"x-lucid-upload": upload})


def resolve_input_file_annotation(annotation: object) -> bool | None:
    if annotation is InputFile:
        return False
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return None
    args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    if len(args) != 1 or args[0] is not InputFile:
        return None
    return True


def _normalize_mime_types(mime_types: Sequence[str] | None) -> tuple[str, ...]:
    if mime_types is None:
        return ()
    normalized = tuple(str(item).strip().lower() for item in mime_types if str(item).strip())
    if not normalized:
        raise ValueError("file_input() requires at least one non-empty MIME type when provided")
    return normalized


def _normalize_max_bytes(max_bytes: int) -> int:
    value = int(max_bytes)
    if value <= 0:
        raise ValueError("file_input() max_bytes must be positive")
    return value
