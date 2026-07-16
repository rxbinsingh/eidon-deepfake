"""
Identity/source grouping for leakage-safe cross-validation.

A random by-video split leaks in this dataset: the face-swaps are generated from
the SAME source clips as the reals (RTFS), and DFD reuses the same actors across
real and manipulated videos. If a source clip's real lands in train and its swap
in test, the model can score by memorizing the shared background/body/compression
rather than the manipulation. `derive_groups` assigns every video a group id such
that all videos sharing an identity/source stay in the same CV fold.

Grouping rules (by filename):
  - RTFS real + inswapper/uniface face-swaps -> group by source clip `{ytid}_{s}_{e}`
    (a real and every swap derived from it share one group).
  - DFD -> union-find over actor ids in `AA_BB__scene__hash` (fake) / `AA__scene`
    (real); group = identity component.
  - FF++ NeuralTextures `TTT_SSS` -> group by target id.
  - Generated (fullbody_gan, t2v_sora, t2v_veo2) -> no identity; each its own group.
"""

import re
from collections import defaultdict
from pathlib import Path

_RTFS_PREFIXES = (
    "rtfs_real_", "week2_real_",
    "rtfs_faceswap_inswapper_", "week2_face_swap_uniface_uniface_",
)
_DFD_RE = re.compile(r"^\d+(_\d+)?__")


def _rtfs_source(stem: str) -> str:
    """Reduce an RTFS real/swap filename stem to its shared source-clip id."""
    s = stem
    for p in _RTFS_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    s = re.sub(r"_\d+-\d+_(man|woman)$", "", s)  # drop swap suffix _{k}-{n}_{gender}
    return s


def derive_groups(samples):
    """samples: iterable of (path_or_name, label, category). Returns list[str] groups."""
    stems = [Path(p).stem for p, _l, _c in samples]

    # Union-find over DFD actor ids so a fake and both its actors' reals coincide.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        parent[find(a)] = find(b)

    for stem in stems:
        if _DFD_RE.match(stem):
            ids = stem.split("__")[0].split("_")
            for i in ids:
                find(i)
            for i in ids[1:]:
                union(ids[0], i)

    groups = []
    for stem in stems:
        low = stem.lower()
        if _DFD_RE.match(stem):
            actor = stem.split("__")[0].split("_")[0]
            groups.append(f"dfd::{find(actor)}")
        elif "neuraltextures" in low or "ffpp" in low:
            m = re.search(r"(\d{3})_(\d{3})", stem)
            groups.append(f"ffpp::{m.group(1) if m else stem}")
        elif ("rtfs_real" in low or "week2_real" in low
              or "faceswap" in low or "face_swap_uniface" in low):
            groups.append(f"rtfs::{_rtfs_source(stem)}")
        else:  # fullbody_gan / t2v_sora / t2v_veo2 — synthetic, no shared identity
            groups.append(f"gen::{stem}")
    return groups


def leakage_report(samples, groups):
    """Summarize how many source-clip groups mix real+fake (the leak a random split hits)."""
    by_group = defaultdict(lambda: {"real": 0, "fake": 0})
    for (_, label, _c), g in zip(samples, groups):
        by_group[g]["real" if label == 0 else "fake"] += 1
    mixed = {g: c for g, c in by_group.items() if c["real"] and c["fake"]}
    leaked_reals = sum(c["real"] for c in mixed.values())
    leaked_fakes = sum(c["fake"] for c in mixed.values())
    return {
        "n_groups": len(by_group),
        "n_mixed_groups": len(mixed),
        "reals_sharing_group_with_fake": leaked_reals,
        "fakes_sharing_group_with_real": leaked_fakes,
    }
