"""Setuptools hook: copy an explicit static-resource allowlist into the wheel.

No experiment data, snapshots, credentials or generated results are packaged.
Retired pre-exp1 code is a separate archive, not part of this distribution.
"""
from pathlib import Path

from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def _static_sources(self):
        root = Path(__file__).resolve().parent
        sources = sorted((root / "knowledge").rglob("*.json"))
        if not sources or any(not path.is_file() for path in sources):
            raise FileNotFoundError("static monitoring configuration/catalogue is missing")
        return root, sources

    def run(self):
        super().run()
        root, sources = self._static_sources()
        destination = Path(self.build_lib) / "paperalpha_monitor" / "_resources"
        for source in sources:
            target = destination / source.relative_to(root)
            self.mkpath(str(target.parent))
            self.copy_file(str(source), str(target))

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        root, sources = self._static_sources()
        destination = Path(self.build_lib) / "paperalpha_monitor" / "_resources"
        return outputs + [str(destination / source.relative_to(root)) for source in sources]

    def get_source_files(self):
        root, sources = self._static_sources()
        extra = [Path(__file__).resolve(), *sources]
        return super().get_source_files() + [str(path.relative_to(root)) for path in extra]
