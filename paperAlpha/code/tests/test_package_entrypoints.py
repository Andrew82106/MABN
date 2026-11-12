"""Packaging/entry contracts only; no API calls or model-quality assertions.

The wheel is built from a small copied tree under pytest's temporary directory,
never in the workspace. The retired pre-exp1 system has its own package and tests in archive/.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
LEGACY_EXPORTS = {
    "EvidenceStream", "TemplateBN", "replay", "DynamicJointBN", "JointRiskStream",
    "TypedJointBN", "TypedRiskStream", "ActionScopedJointBN", "ActionRiskStream",
    "CapabilityActionBN", "CapabilityRiskStream", "CompactCapabilityBN",
    "CompactCapabilityStream",
}
CORE_EXPORTS = {
    "MessageWorkflowCompiler", "ObservableEventDAG", "ObservableEventLaw",
    "DAGWorkflowInput", "DAGWorkflowRiskBN", "EliminatingEventLaw",
    "EliminatingDAGWorkflowRiskBN", "WorkflowRiskInput", "WorkflowRiskBN",
    "WorkflowFragmentRegistry", "AuthorityProofGraph", "AuthorityWindow",
    "bind_message_policy", "GroupedEventTrace", "ForecastAlarmStream",
}


def _run(args, *, cwd):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _python(code, *, cwd, paths):
    prefix = f"import sys; sys.path[:0] = {list(map(str, paths))!r}\n"
    return _run([sys.executable, "-I", "-B", "-c", prefix + textwrap.dedent(code)], cwd=cwd)


def test_package_import_is_lazy_and_configuration_free(tmp_path):
    result = _python("""
        import builtins
        from pathlib import Path
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'numpy', 'scipy', 'sklearn'}:
                raise AssertionError('package import eagerly loaded ' + name)
            return original_import(name, *args, **kwargs)
        def no_static_read(*args, **kwargs):
            raise AssertionError('package import read a file')
        builtins.__import__ = guarded_import
        Path.read_text = no_static_read
        Path.read_bytes = no_static_read
        import paperalpha_monitor as monitor
        import paperalpha_runtime
        assert not any(n.startswith('paperalpha_monitor.') for n in sys.modules)
        assert 'MessageWorkflowCompiler' in dir(monitor)
        try:
            monitor.not_a_model
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown export must fail normally')
        print('lazy configuration-free package import')
    """, cwd=tmp_path, paths=[ROOT / 'code/src'])
    assert "lazy configuration-free" in result.stdout


def test_current_and_legacy_exports_resolve_to_original_objects():
    import paperalpha_monitor as monitor
    assert set(monitor.__all__) == CORE_EXPORTS | LEGACY_EXPORTS
    for name in monitor.__all__:
        original = getattr(importlib.import_module(
            f"paperalpha_monitor.{monitor._EXPORTS[name]}"), name)
        assert getattr(monitor, name) is original
        assert monitor.__dict__[name] is original


def test_source_resources_keep_original_paths_and_bytes():
    from paperalpha_monitor._resources import static_resource
    for relative in ['knowledge/risks/risk_catalog.json', *(
            path.relative_to(ROOT).as_posix() for path in (ROOT / "knowledge").rglob("*.json"))]:
        resolved = static_resource(relative)
        assert resolved == ROOT / relative
        assert resolved.read_bytes() == (ROOT / relative).read_bytes()


@pytest.mark.parametrize("name", ["../.env", "/configs/x.json", 'knowledge/templates/../x.json',
                                 "configs\\x.json", "results/x.json", "knowledge/private/x.json"])
def test_resource_locator_rejects_nonstatic_paths(name):
    from paperalpha_monitor._resources import static_resource
    with pytest.raises(ValueError):
        static_resource(name)


def test_resource_locator_does_not_fall_back_to_working_directory(tmp_path, monkeypatch):
    from paperalpha_monitor._resources import static_resource
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        static_resource('knowledge/templates/nonexistent-packaging-test.json')


def test_pytest_and_package_entries_cover_current_and_legacy_code():
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(settings["tool"]["setuptools"]["packages"]) == {
        "paperalpha_monitor", "paperalpha_runtime"}
    assert set(settings["tool"]["pytest"]["ini_options"]["testpaths"]) == {
        "code/tests"}
    assert "pre-exp1" not in settings["project"]["scripts"]
    assert settings["project"]["scripts"]["paperalpha-evaluate"] == "paperalpha_runtime.cli:main"
    assert settings["project"]["name"] == "paperalpha"


def test_default_suite_is_current_and_legacy_remains_separate():
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    roots = [ROOT / p for p in settings["tool"]["pytest"]["ini_options"]["testpaths"]]
    assert roots == [ROOT / "code/tests"]
    assert len(list(roots[0].rglob("test_*.py"))) > 100
    archived = ROOT / "archive/early_pre_exp1"
    old = tomllib.loads((archived / "pyproject.toml").read_text(encoding="utf-8"))
    assert old["project"]["scripts"]["pre-exp1"] == "pre_exp1.cli:main"
    assert (archived / old["tool"]["pytest"]["ini_options"]["testpaths"][0]).is_dir()


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline-wheel")
    stage = root / "source"
    stage.mkdir()
    for name in ("pyproject.toml", "README.md", "packaging_build.py"):
        shutil.copyfile(ROOT / name, stage / name)
    for name in ("code/src", "knowledge"):
        shutil.copytree(ROOT / name, stage / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    wheels = root / "wheels"
    _run([sys.executable, "-B", "-m", "pip", "wheel", "--no-index", "--no-deps",
          "--no-build-isolation", "--no-cache-dir", "--wheel-dir", str(wheels), str(stage)], cwd=root)
    candidates = list(wheels.glob("*.whl"))
    assert len(candidates) == 1
    return root, candidates[0]


def test_offline_wheel_contains_all_packages_and_only_static_resources(built_wheel):
    _, wheel = built_wheel
    with zipfile.ZipFile(wheel) as package:
        names = package.namelist()
        for root in ("paperalpha_monitor", "paperalpha_runtime"):
            assert f"{root}/__init__.py" in names
        for source in (ROOT / 'code/src').glob("*/*.py"):
            assert source.relative_to(ROOT / 'code/src').as_posix() in names
        resources = ['knowledge/risks/risk_catalog.json', *(
            path.relative_to(ROOT).as_posix() for path in (ROOT / "knowledge").rglob("*.json"))]
        for relative in resources:
            assert package.read(f"paperalpha_monitor/_resources/{relative}") == (ROOT / relative).read_bytes()
        assert all(not any(part in {"results", "experiments", "archive", "data", ".env", "tests", ".local", "external"}
                           for part in Path(name).parts) for name in names)
        assert all(not name.endswith((".pyc", ".pyo")) for name in names)
        metadata = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
        assert "paperalpha-evaluate = paperalpha_runtime.cli:main" in package.read(metadata).decode()
        assert "pre-exp1" not in package.read(metadata).decode()


def test_installed_wheel_instantiates_models_outside_checkout(built_wheel):
    root, wheel = built_wheel
    target, outside = root / "installed", root / "outside"
    outside.mkdir()
    _run([sys.executable, "-B", "-m", "pip", "install", "--no-index", "--no-deps",
          "--no-compile", "--no-cache-dir", "--target", str(target), str(wheel)], cwd=outside)
    result = _python("\n        import json\n        from pathlib import Path\n        import paperalpha_monitor as monitor\n        import paperalpha_runtime\n        install = Path(sys.path[0]).resolve()\n        for module in (monitor, paperalpha_runtime):\n            assert Path(module.__file__).resolve().is_relative_to(install)\n        assert not any(n.startswith('paperalpha_monitor.') for n in sys.modules)\n        # Every compatibility name resolves, without falling back to checkout code.\n        for name in monitor.__all__:\n            assert getattr(monitor, name) is not None\n        compiler = monitor.MessageWorkflowCompiler()\n        law = monitor.ObservableEventLaw()\n        eliminating = monitor.EliminatingDAGWorkflowRiskBN()\n        ordinary = monitor.DAGWorkflowRiskBN()\n        policy = monitor.WorkflowRiskBN()\n        assert len(law.names) == 25\n        assert len(eliminating.names) == len(ordinary.names) == 28\n        assert eliminating.parameters()['inference_algorithm'] == 'exact_log_score_variable_elimination'\n        assert compiler.templates and policy.registry.templates\n        from paperalpha_monitor._resources import static_resource\n        assert static_resource('knowledge/risks/risk_catalog.json').is_relative_to(install)\n        from paperalpha_monitor.choice_attribute_bn import AttributeChoiceWorkflowRiskBN\n        from paperalpha_monitor.privacy_flow_bn import PrivacyFlowBN\n        assert AttributeChoiceWorkflowRiskBN().names and PrivacyFlowBN().names\n        for name, module in list(sys.modules.items()):\n            if name.startswith(('paperalpha_monitor', 'paperalpha_runtime')):\n                assert Path(module.__file__).resolve().is_relative_to(install), name\n        print(json.dumps({'installed_root': str(install), 'core_parameters': len(eliminating.names),\n                          'exports': len(monitor.__all__)}))\n    ", cwd=outside, paths=[target])
    assert json.loads(result.stdout)["core_parameters"] == 28
