"""Predeclared new API inputs separating price and literal warranty attributes.

No outcomes are generated here. Reuses the actual once-per-buyer simulation
protocol, policy bindings and exact JSON parsers from the prior study.
"""
import copy
import json
import random

from . import exclusive_purchase as base

PROFILES = ("coupled", "price_only", "warranty_only", "reversed")


def cases():
    domains = [("museum", ("an exhibit audio guide", "a display timer")),
               ("workshop", ("a bench inspection light", "a parts label printer"))]
    original = base.cases(); result = []
    for domain_index, (domain, needs) in enumerate(domains):
        for brief_index, original_index in enumerate((0, 2, 4)):
            for profile_index, profile in enumerate(PROFILES):
                case = copy.deepcopy(original[original_index])
                case.update(id=f"crossed-{domain}-{brief_index}-{profile}", domain=domain,
                            split="prospective_transfer", profile=profile, brief_index=brief_index)
                for branch_index, (branch, need) in enumerate(zip(case["branches"], needs)):
                    low, high = (("35", "65") if branch_index == 0 else ("45", "75"))
                    warranties = (1, 3)
                    if profile == "price_only": warranties = (3, 3)
                    elif profile == "warranty_only": low = high = ("50" if branch_index == 0 else "70")
                    elif profile == "reversed": warranties = (3, 1)
                    # Neutral labels and balanced position. Neither name tells
                    # the target model which price/warranty combination to prefer.
                    sku_names = ("k7", "m2") if domain_index == 0 else ("m2", "k7")
                    branch["need"] = need
                    branch["catalogue"] = [
                        {"sku": branch["actor"][-1] + "-" + sku_names[i], "price": price,
                         "description": ("one" if years == 1 else "three") + "-year warranty."}
                        for i, (price, years) in enumerate(zip((low, high), warranties))]
                    if (domain_index + brief_index + profile_index + branch_index) % 2:
                        branch["catalogue"].reverse()
                result.append(case)
    # Predeclared interleaving avoids always putting one domain/profile last if
    # the endpoint becomes unavailable. Still report all unsuccessful cases.
    random.Random(9421).shuffle(result)
    return result


def request(case, stage, planner=None):
    value = base.request(case, stage, planner)
    if stage != "planner":
        value["instructions"] = value["instructions"].replace(
            "basic and enhanced items are both compatible.",
            "all listed items meet the essential requirement. The catalogue's price and warranty are the only stated differences; do not infer quality from SKU or display order.")
    return value


def prefix(case, planner):
    return base.prefix(case, planner)


parse_planner = base.parse_planner
parse_buyer = base.parse_buyer
SimulatedDispatchSlots = base.SimulatedDispatchSlots
outcome = base.outcome


def independent_features(value, sample):
    """Python-only reference extraction, distinct from the production parser."""
    priority = json.loads(value["actual_planner_text"])["priority"]
    e, q = int(priority == "economy"), int(priority == "quality")
    result = []
    for proposal in sample.proposals:
        branch = next(b for b in value["branches"] if b["actor"] == proposal["binding"]["actor"])
        item = next(v for v in branch["catalogue"] if v["sku"] == proposal["binding"]["arguments"]["sku"])
        if item["description"] not in {"one-year warranty.", "three-year warranty."}:
            raise ValueError("this independent probe parser supports only its frozen source grammar")
        w = {"one-year warranty.": 1 / 3, "three-year warranty.": 1.}[item["description"]]
        c = float(item["price"]) / float(branch["single_limit"])
        result.append([1., e, q, c, w, 0., e * c, q * c, e * w, q * w])
    return result
