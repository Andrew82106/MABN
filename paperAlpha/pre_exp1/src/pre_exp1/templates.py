"""Frozen and hash-verified P0 message templates."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import stable_hash


class TemplateConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenTemplate:
    template_id: str
    version: str
    content: dict[str, Any]
    content_hash: str


class TemplateRegistry:
    def __init__(
        self,
        *,
        source_path: Path,
        version: str,
        templates: dict[str, FrozenTemplate],
    ) -> None:
        self.source_path = source_path.resolve()
        self.version = version
        self._templates = templates

    @classmethod
    def from_file(cls, path: Path) -> "TemplateRegistry":
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("hash_method") != "sha256-canonical-json":
            raise TemplateConfigError("Unsupported message template hash method")
        templates: dict[str, FrozenTemplate] = {}
        for key, value in config.get("templates", {}).items():
            template_id = value.get("template_id")
            if not template_id or template_id in templates:
                raise TemplateConfigError("Template IDs must be non-empty and unique")
            if key != template_id:
                raise TemplateConfigError(
                    f"Template key {key!r} must equal template_id {template_id!r}"
                )
            content = copy.deepcopy(value["content"])
            computed = stable_hash(content)
            declared = value.get("content_hash")
            if declared != computed:
                raise TemplateConfigError(
                    f"Template {template_id!r} hash mismatch: "
                    f"declared {declared!r}, computed {computed!r}"
                )
            templates[template_id] = FrozenTemplate(
                template_id=template_id,
                version=str(value["version"]),
                content=content,
                content_hash=computed,
            )
        required = {"normal_original", "dangerous_original", "safe", "drop"}
        if set(templates) != required:
            raise TemplateConfigError(
                f"Expected exactly the frozen templates {sorted(required)}"
            )
        return cls(
            source_path=path,
            version=str(config["version"]),
            templates=templates,
        )

    def get(self, template_id: str) -> FrozenTemplate:
        try:
            template = self._templates[template_id]
        except KeyError as exc:
            raise TemplateConfigError(
                f"Unknown frozen template {template_id!r}"
            ) from exc
        return FrozenTemplate(
            template_id=template.template_id,
            version=template.version,
            content=copy.deepcopy(template.content),
            content_hash=template.content_hash,
        )

    def hashes(self) -> dict[str, str]:
        return {
            key: template.content_hash
            for key, template in sorted(self._templates.items())
        }

