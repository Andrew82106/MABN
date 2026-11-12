from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT / 'code/src'),str(ROOT / 'code/commands')]
from run_forecast_alarm_study import split_templates,bind_partition
from verify_forecast_alarm_study import independent_threshold
from paperalpha_monitor.alarm_policy import calibrate_phase


def test_template_split_is_label_blind_and_keeps_groups_together():
    rows=[{"id":str(i),"template_id":"group-"+str(i//3),"y":i%2} for i in range(24)]
    config={"split_seed":"fixed-label-blind-seed:","calibration_template_count":2}
    a,ordered=split_templates(rows,config)
    b,other=split_templates([{**r,"y":1-r["y"]} for r in reversed(rows)],config)
    assert a==b and ordered==other and sum(p=="calibration" for p in a.values())==6
    for template in ordered:
        assert len({a[r["id"]] for r in rows if r["template_id"]==template})==1


def test_partition_rejects_template_overlap_and_parent_boundary_changes():
    rows=[{"id":"a","template_id":"one","split":"train"},{"id":"b","template_id":"two","split":"train"},
        {"id":"c","template_id":"three","split":"validation"}]
    plan={"partition_by_id":{"a":"fit","b":"calibration","c":"validation"},"partition_counts":{"fit":1,"calibration":1,"validation":1}}
    assert all(len(v)==1 for v in bind_partition(rows,plan).values())
    with pytest.raises(ValueError,match="contamination"):
        bind_partition([rows[0],{**rows[1],"template_id":"one"},rows[2]],plan)
    with pytest.raises(ValueError,match="boundary"):
        bind_partition([rows[0],rows[1],{**rows[2],"split":"train"}],plan)


@pytest.mark.parametrize("values,budget",[([.2,.2,.5,.9],.25),([1.,1.,1.],.1),([0.,0.,.1],0.),([.1,.9],1.)])
def test_independent_threshold_reproduces_ties(values,budget):
    assert independent_threshold(values,budget)==calibrate_phase(values,budget)["threshold"]
