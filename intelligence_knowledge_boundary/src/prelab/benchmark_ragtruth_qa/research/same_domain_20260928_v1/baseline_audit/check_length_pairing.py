"""CPU-only audit of optional within-update padding reduction."""
import random
from pathlib import Path
import ragognizer_same_domain as main
from ragognizer_batch_runtime import pair_inside_updates


def main_check():
    fit = main.rows(main.HERE / "ragognizer_prepared_v2/fit.jsonl")
    lengths = [len(row["input_ids"]) for row in fit]
    order = list(range(len(fit)))
    reports = []
    for epoch in range(1, 6):
        random.Random(20260928 + epoch).shuffle(order)
        before = list(order)
        paired = pair_inside_updates(order, lengths)
        assert order == before, "Pairing must not change future epoch shuffle state"
        for start in range(0, len(order), 8):
            assert set(order[start:start+8]) == set(paired[start:start+8])
        def cost(indices, power=1):
            return sum(len(indices[i:i+2]) * max(lengths[j] for j in indices[i:i+2]) ** power
                       for i in range(0, len(indices), 2))
        plain, reduced = cost(order), cost(paired)
        assert reduced <= plain
        assert cost(paired, 2) <= cost(order, 2)
        reports.append({"epoch": epoch, "updates": (len(order)+7)//8,
                        "valid_tokens": sum(lengths), "unpaired_padded_tokens": plain,
                        "paired_padded_tokens": reduced,
                        "padded_token_reduction_fraction": 1-reduced/plain,
                        "unchanged_update_members": True, "unmodified_shuffle_state": True})
    result = {"status": "PASS", "fit_answers": len(fit), "epochs": reports,
              "effective_answers_per_update": 8, "gpu_used": False, "test_opened": False,
              "training_code_sha256": main.sha(main.__file__),
              "batch_runtime_sha256": main.sha(Path(__file__).with_name("ragognizer_batch_runtime.py"))}
    main.save(main.HERE / "cpu_checks/LENGTH_PAIRING_CHECK.json", result)
    print(result)


if __name__ == "__main__":
    main_check()
