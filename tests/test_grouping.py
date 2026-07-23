"""Tests for src/grouping.py identity/source grouping.

Runnable directly (`python tests/test_grouping.py`) or under pytest.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grouping import derive_groups, leakage_report


def _samples():
    # (filename, label, category) — mirrors real naming conventions
    return [
        ("rtfs_real_bdDhQL1aJcM_00008_00013.mp4", 0, "real"),
        ("rtfs_faceswap_inswapper_bdDhQL1aJcM_00008_00013_1-35_woman.mp4", 1, "face_swap"),
        ("rtfs_faceswap_inswapper_bdDhQL1aJcM_00008_00013_3-98_woman.mp4", 1, "face_swap"),
        ("week2_real_d6C4_HZzYbk_00487_00500.mp4", 0, "real"),
        ("week2_face_swap_uniface_uniface_d6C4_HZzYbk_00487_00500_3-1_man.mp4", 1, "face_swap_uniface"),
        ("ffpp_neuraltextures_001_870.mp4", 1, "expression_swap"),
        ("ffpp_neuraltextures_001_022.mp4", 1, "expression_swap"),
        ("18__exit_phone_room.mp4", 0, "real"),
        ("06_20__walk_down_hall_angry__6SUW7063.mp4", 1, "face_swap"),
        ("20__secret_conversation.mp4", 0, "real"),
        ("fakeparts_fullbody_t2v_sora-720p_0453.mp4", 1, "fullbody_gan"),
        ("week2_t2v_sora_1080p_t2v_sora-1080p_0000.mp4", 1, "t2v_sora_1080p"),
    ]


def test_rtfs_real_and_swaps_share_group():
    s = _samples()
    g = derive_groups(s)
    gm = dict(zip([x[0] for x in s], g))
    # a real clip and every swap derived from it must be one group
    real = gm["rtfs_real_bdDhQL1aJcM_00008_00013.mp4"]
    assert gm["rtfs_faceswap_inswapper_bdDhQL1aJcM_00008_00013_1-35_woman.mp4"] == real
    assert gm["rtfs_faceswap_inswapper_bdDhQL1aJcM_00008_00013_3-98_woman.mp4"] == real
    # week2 real + its uniface swap share a group
    assert gm["week2_real_d6C4_HZzYbk_00487_00500.mp4"] == \
        gm["week2_face_swap_uniface_uniface_d6C4_HZzYbk_00487_00500_3-1_man.mp4"]


def test_dfd_actors_union_find():
    s = _samples()
    g = derive_groups(s)
    gm = dict(zip([x[0] for x in s], g))
    # fake 06_20 links actors 06 and 20; real 20 must join that component
    assert gm["06_20__walk_down_hall_angry__6SUW7063.mp4"] == gm["20__secret_conversation.mp4"]
    # a different actor (18) is its own component
    assert gm["18__exit_phone_room.mp4"] != gm["20__secret_conversation.mp4"]


def test_ffpp_and_generated_grouping():
    s = _samples()
    g = derive_groups(s)
    gm = dict(zip([x[0] for x in s], g))
    # FF++ grouped by target id -> both 001_* share a group
    assert gm["ffpp_neuraltextures_001_870.mp4"] == gm["ffpp_neuraltextures_001_022.mp4"]
    # synthetic t2v/fullbody are each their own group (no shared identity)
    assert gm["fakeparts_fullbody_t2v_sora-720p_0453.mp4"].startswith("gen::")
    assert gm["week2_t2v_sora_1080p_t2v_sora-1080p_0000.mp4"].startswith("gen::")


def test_leakage_report_counts_mixed_groups():
    s = _samples()
    g = derive_groups(s)
    rep = leakage_report(s, g)
    # RTFS + DFD groups mix real and fake; report must flag them
    assert rep["n_mixed_groups"] >= 3
    assert rep["fakes_sharing_group_with_real"] >= 3
    assert rep["n_groups"] == len(set(g))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all grouping tests passed")
