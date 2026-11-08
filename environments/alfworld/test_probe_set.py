"""training-probe Phase 0 -- probe-set loader and the frozen alfworld-24-v1 set.

Pure stdlib (probe_sets.py imports nothing from verifiers / alfworld), so this
runs anywhere. The selection-script cross-check runs only when the thesis
script is reachable (env var PROBE_SET_SELECT_DIR, or the default WSL mount
path); otherwise it is skipped, not failed.

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python -m pytest environments/alfworld/test_probe_set.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.alfworld.probe_sets import (  # noqa: E402
    PROBE_SETS_DIR,
    list_probe_sets,
    load_probe_set,
    resolve_game_file,
    validate_probe_games,
)

TASK_TYPES = {
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
}

DEFAULT_SELECT_DIR = Path(
    "/mnt/c/Users/SBC98/Desktop/Bocconi/5-2026-spring/thesis/cluster-files/probe-set"
)


def test_alfworld_24_v1_is_listed():
    assert "alfworld-24-v1" in list_probe_sets()
    assert (PROBE_SETS_DIR / "alfworld-24-v1.json").is_file()


def test_probe_set_loads_24_unique():
    games = load_probe_set("alfworld-24-v1")
    assert len(games) == 24
    assert len({g["probe_id"] for g in games}) == 24
    assert len({g["game_id"] for g in games}) == 24
    # 6 task types x 2 splits x 2 roles, every cell filled exactly once
    cells = {(g["split"], g["task_type"], g["role"]) for g in games}
    assert len(cells) == 24
    assert {g["task_type"] for g in games} == TASK_TYPES
    assert {g["split"] for g in games} == {"valid_seen", "valid_unseen"}
    assert {g["role"] for g in games} == {"bite", "control"}


def test_probe_ids_follow_scheme():
    games = load_probe_set("alfworld-24-v1")
    for g in games:
        split_tag, type_tag, role = g["probe_id"].split("-")
        assert split_tag == ("seen" if g["split"] == "valid_seen" else "unseen")
        assert role == g["role"]
        assert type_tag in {"look", "pps", "clean", "cool", "heat", "two"}
        # game_id's task dir starts with the task type
        assert g["game_id"].split("/")[1].startswith(g["task_type"])


def test_bite_and_control_in_distinct_task_dirs():
    games = load_probe_set("alfworld-24-v1")
    by_cell = {}
    for g in games:
        by_cell.setdefault((g["split"], g["task_type"]), {})[g["role"]] = g
    for cell, roles in by_cell.items():
        bite_dir = roles["bite"]["game_id"].split("/")[1]
        ctrl_dir = roles["control"]["game_id"].split("/")[1]
        assert bite_dir != ctrl_dir, cell
        # selection thresholds (D-5)
        assert roles["bite"]["long_pass1"] >= 0.75
        assert roles["control"]["long_pass1"] >= 0.875
        assert roles["control"]["short_pass1"] >= 0.875


def test_unknown_set_raises_with_available_list():
    with pytest.raises(FileNotFoundError) as exc:
        load_probe_set("does-not-exist")
    assert "alfworld-24-v1" in str(exc.value)


def test_validate_rejects_duplicates_and_bad_fields():
    good = {
        "probe_id": "seen-look-bite",
        "game_id": "valid_seen/look_at_obj_in_light-X/trial_1",
        "split": "valid_seen",
        "task_type": "look_at_obj_in_light",
        "role": "bite",
    }
    validate_probe_games([good])
    with pytest.raises(ValueError, match="duplicate probe_id"):
        validate_probe_games([good, dict(good, game_id="valid_seen/other/trial_2")])
    with pytest.raises(ValueError, match="duplicate game_id"):
        validate_probe_games([good, dict(good, probe_id="seen-look-control", role="control")])
    with pytest.raises(ValueError, match="role"):
        validate_probe_games([dict(good, role="neither")])
    with pytest.raises(ValueError, match="starts with"):
        validate_probe_games([dict(good, game_id="valid_unseen/look_at_obj_in_light-X/trial_1")])
    with pytest.raises(ValueError, match="missing keys"):
        validate_probe_games([{k: v for k, v in good.items() if k != "role"}])


def test_resolve_game_file_joins_under_data_path():
    p = resolve_game_file("/data/json_2.1.1", "valid_seen/task-dir/trial_1")
    assert p.replace("\\", "/") == "/data/json_2.1.1/valid_seen/task-dir/trial_1/game.tw-pddl"


def test_probe_set_matches_selection_script(tmp_path):
    """Rerun the thesis selection script on the frozen CSV; `games` must match."""
    select_dir = Path(os.environ.get("PROBE_SET_SELECT_DIR", DEFAULT_SELECT_DIR))
    script = select_dir / "select_probe_games.py"
    csv_path = select_dir / "per_problem.csv"
    if not (script.is_file() and csv_path.is_file()):
        pytest.skip(f"selection script not reachable at {select_dir}")
    out = tmp_path / "fresh.json"
    subprocess.run(
        [sys.executable, "-I", str(script), "--per-problem", str(csv_path), "--out", str(out), "--created", "x"],
        check=True,
        capture_output=True,
    )
    fresh = json.loads(out.read_text(encoding="utf-8"))["games"]
    committed = load_probe_set("alfworld-24-v1")
    assert fresh == committed
