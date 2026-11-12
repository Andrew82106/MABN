import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/harnessaudit_cohort.py'
spec = importlib.util.spec_from_file_location("cohort_test", SCRIPT)
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


def candidates():
    return [{"task_id": f"dl-t{10 * i + j}", "category": category, "eligible": True}
            for i, category in enumerate(cohort.CATEGORIES, 1) for j in (1, 2, 3)]


def test_selection_is_order_independent_balanced_and_not_outcome_driven():
    source = candidates()
    expected = cohort.select_tasks(source)
    assert len(expected) == 4 and {t["category"] for t in expected} == set(cohort.CATEGORIES)
    assert cohort.select_tasks(source[::-1]) == expected
    changed = copy.deepcopy(source)
    for n, item in enumerate(changed):
        item.update(label=n % 2, score=n / 20, task_sha256=str(n))
    assert [t["task_id"] for t in cohort.select_tasks(changed)] == [t["task_id"] for t in expected]


def test_exclusions_and_insufficient_category_do_not_trigger_replacement():
    source = candidates()
    for item in source:
        if item["category"] == "housing":
            item["eligible"] = False
    with pytest.raises(ValueError, match="insufficient"):
        cohort.select_tasks(source)


@pytest.mark.parametrize("count", [0, 3, True, 1.5])
def test_bounded_selection(count):
    with pytest.raises(ValueError):
        cohort.select_tasks(candidates(), count)


def test_inventory_excludes_previously_seen_and_multimodal(tmp_path):
    root = tmp_path / "multi_agent/tasks/daily_life/dining"
    root.mkdir(parents=True)
    for identity, modality in [("dl-t11", "text_only"), ("dl-t18", "text_only"), ("dl-t22", "multimodal")]:
        cohort.write_new(root / f"{identity}.yaml", {"task_id": identity, "category": "dining", "domain": "daily_life", "modality": modality})
    records = {t["task_id"]: t for t in cohort.inventory(tmp_path)}
    assert not records["dl-t11"]["eligible"] and not records["dl-t22"]["eligible"]
    assert records["dl-t18"]["eligible"]
    assert set(records["dl-t18"]) == {"task_id", "category", "task_path", "task_sha256", "eligible", "exclusion_reasons"}


def test_plan_edit_and_duplicate_writes_rejected(tmp_path):
    plan = {"schema": "harnessaudit-development-cohort-v1", "settings": {"model": "gpt-5.5"}}
    plan["plan_sha256"] = cohort.digest(plan)
    path = tmp_path / "PLAN.json"
    cohort.write_new(path, plan)
    assert cohort.load_plan(path, verify_sources=False) == plan
    with pytest.raises(FileExistsError):
        cohort.write_new(path, plan)
    plan["settings"]["model"] = "different"
    changed = tmp_path / "edited.json"
    cohort.write_new(changed, plan)
    with pytest.raises(ValueError, match="changed"):
        cohort.load_plan(changed, verify_sources=False)


def test_real_metadata_selection_does_not_include_previous_tasks():
    selected = cohort.select_tasks(cohort.inventory(cohort.UPSTREAM))
    assert len(selected) == 4
    assert not set(cohort.EXCLUDED) & {t["task_id"] for t in selected}
