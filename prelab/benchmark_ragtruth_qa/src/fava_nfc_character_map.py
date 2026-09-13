"""Strict NFC composition coverage repair for auxiliary FAVA only.

Original text, offsets and labels are not changed. No label is accepted here.
Unproven gaps fail. Complete original coverage takes the exact old QA path.
"""
from collections import Counter
import json
import unicodedata as ud
import numpy as np
from tokenizers import NormalizedString
import tail_finetune as old_mapping


def _require(condition, message):
    if not condition: raise ValueError(message)


def _proof(text, missing, normalizer):
    """Prove a missing mark was consumed into one NFC output character."""
    _require(ud.category(text[missing]).startswith('M') and ud.combining(text[missing]) > 0,
             f'Unproven NFC gap at {missing}: not a positive-class combining mark')
    left = missing
    while left > 0 and ud.combining(text[left]) > 0: left -= 1
    _require(ud.combining(text[left]) == 0 and not ud.category(text[left]).startswith('M')
             and not text[left].isspace(), f'No covered NFC starter for gap {missing}')
    right = left+1
    while right < len(text) and ud.combining(text[right]) > 0: right += 1
    cluster = text[left:right]
    aligned = NormalizedString(cluster)
    normalizer.normalize(aligned)
    composed = aligned.normalized
    _require(composed == ud.normalize('NFC', cluster), f'NFC implementation disagreement at {missing}')
    _require(len(composed) == 1 and composed != cluster,
             f'Gap {missing} is not proven single-character NFC composition')
    _require(ud.normalize('NFD', composed) == ud.normalize('NFD', cluster),
             f'Canonical equivalence not established at {missing}')
    removed = text[left:missing]+text[missing+1:right]
    _require(normalizer.normalize_str(removed) != composed,
             f'Gap {missing} not needed for this NFC result')
    # Current HF source alignment anchors a composed character to its starter.
    # Require the actual runtime behavior instead of assuming it from a version.
    source_anchor = aligned[0:1].original
    _require(aligned.original == cluster and source_anchor == text[left:left+1],
             f'Unexpected NormalizedString source anchor at {missing}')
    return {'missing_character_index': missing, 'missing_codepoint': f'U+{ord(text[missing]):04X}',
        'cluster_start': left, 'cluster_end': right, 'original_cluster': cluster,
        'normalized_character': composed, 'normalized_codepoint': f'U+{ord(composed):04X}',
        'normalized_string_source_anchor': source_anchor, 'starter_index': left,
        'canonical_equivalence': True, 'removing_mark_changes_NFC': True}


def character_map(text, raw_offsets, start, end, *, normalizer, return_diagnostics=False):
    """Return old-style (raw rows, encoder columns, FP32 weights).

    Pass bert.backend_tokenizer.normalizer. Only the exact NFC normalizer is
    accepted. With return_diagnostics=True return (mapping, proof dictionary).
    The caller should preserve each proof with the auxiliary row provenance.
    """
    state = normalizer.__getstate__()
    if isinstance(state, bytes): state = state.decode('utf-8')
    _require(json.loads(state) == {'type': 'NFC'}, 'FAVA helper requires exactly NFC, not a pipeline or accent stripping')
    _require(len(start) == len(end), 'Encoder offset length mismatch')
    owners = [[] for _ in text]
    for j, (a, b) in enumerate(zip(start, end)):
        a, b = int(a), int(b)
        if a < 0 or b < 0:
            _require(a == b == -1, 'Malformed non-answer encoder boundary'); continue
        _require(0 <= a < b <= len(text), 'Encoder boundary outside original answer')
        for c in range(a, b):
            if not text[c].isspace(): owners[c].append(j)
    missing = [c for c, ch in enumerate(text) if not ch.isspace() and not owners[c]]
    if not missing:
        result = old_mapping.character_map(text, raw_offsets, start, end)
        diagnostics = {'used_legacy_path': True, 'repaired_character_count': 0, 'repairs': [],
                       'original_text_and_offsets_changed': False}
        return (result, diagnostics) if return_diagnostics else result
    repairs = []
    for c in missing:
        proof = _proof(text, c, normalizer)
        anchor = proof['starter_index']
        _require(bool(owners[anchor]), f'No actual encoder token owns NFC starter at {anchor}')
        # Do not accept mixed ownership inside a composed character.
        for k in range(proof['cluster_start'], proof['cluster_end']):
            _require(not owners[k] or owners[k] == owners[anchor],
                     f'Ambiguous encoder ownership within NFC composition at {c}')
        owners[c] = owners[anchor].copy()
        proof['encoder_token_indices'] = owners[c].copy()
        repairs.append(proof)
    _require(all(owners[c] for c, ch in enumerate(text) if not ch.isspace()),
             'Missing full-answer nonwhitespace coverage after proven NFC repairs')
    # Same original-character mass and duplicate-owner sharing as the old map.
    rows, columns, weights = [], [], []
    for i, (a,b) in enumerate(raw_offsets):
        a, b = int(a), int(b)
        _require(0 <= a <= b <= len(text), 'Raw boundary outside original answer')
        chars = [c for c in range(a,b) if not text[c].isspace()]
        counts = Counter()
        for c in chars:
            for j in owners[c]: counts[j] += 1/(len(chars)*len(owners[c]))
        if chars: _require(abs(sum(counts.values())-1) < 1e-12, 'Raw token mass is not one')
        for j, w in sorted(counts.items()): rows.append(i); columns.append(j); weights.append(w)
    result = np.asarray(rows, np.int64), np.asarray(columns, np.int64), np.asarray(weights, np.float32)
    diagnostics = {'used_legacy_path': False, 'repaired_character_count': len(repairs),
        'repairs': repairs, 'original_text_and_offsets_changed': False,
        'scope': 'Only proven single-character NFC compositions; other gaps fail'}
    return (result, diagnostics) if return_diagnostics else result
