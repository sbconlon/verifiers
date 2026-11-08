"""Probe-set loader for the training-time probe (training-probe plan, Phase 0).

A probe set is a frozen, versioned list of ALFWorld games that the orchestrator
replays against the live policy every N training steps. Sets ship as JSON next
to this module under ``probe_sets/<name>.json`` so the list travels with the env
code that consumes it (master doc D-5): a run config names the set, it cannot
drift from it.

Pure stdlib. Importable without verifiers / alfworld / datasets, so the loader
and its tests run anywhere. ``alfworld_env.build_dataset`` consumes
``load_probe_set`` + ``resolve_game_file`` from Phase 1 on.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROBE_SETS_DIR = Path(__file__).resolve().parent / "probe_sets"

REQUIRED_GAME_KEYS = ("probe_id", "game_id", "split", "task_type", "role")
ROLES = ("bite", "control")
SPLITS = ("valid_seen", "valid_unseen", "train")
GAME_FILE_NAME = "game.tw-pddl"


def list_probe_sets() -> list[str]:
    """Names of the probe sets shipped next to this module (sorted)."""
    if not PROBE_SETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROBE_SETS_DIR.glob("*.json"))


def load_probe_set(name: str) -> list[dict]:
    """Return the validated ``games`` list of probe set ``name``.

    Each game dict carries at least ``probe_id``, ``game_id`` (relative to the
    ALFWorld data root, e.g. ``valid_seen/<task-dir>/<trial>``), ``split``,
    ``task_type`` and ``role`` (``bite`` | ``control``). Order is the file's
    order and is preserved by callers (the probe dataset is built in it).
    """
    path = PROBE_SETS_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"probe set {name!r} not found at {path}; available: {list_probe_sets()}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    games = data.get("games") if isinstance(data, dict) else None
    if not isinstance(games, list) or not games:
        raise ValueError(f"probe set {name!r} has no non-empty 'games' list")
    validate_probe_games(games)
    return games


def validate_probe_games(games: list[dict]) -> None:
    """Structural checks shared by the loader and the selection script.

    Raises ValueError on: missing keys, duplicate probe_id / game_id, unknown
    role or split, or a game_id whose first path component disagrees with its
    declared split.
    """
    seen_probe_ids: set[str] = set()
    seen_game_ids: set[str] = set()
    for i, g in enumerate(games):
        if not isinstance(g, dict):
            raise ValueError(f"game #{i} is not an object")
        missing = [k for k in REQUIRED_GAME_KEYS if k not in g]
        if missing:
            raise ValueError(f"game #{i} missing keys {missing}")
        if g["role"] not in ROLES:
            raise ValueError(f"game {g['probe_id']!r}: role {g['role']!r} not in {ROLES}")
        if g["split"] not in SPLITS:
            raise ValueError(f"game {g['probe_id']!r}: split {g['split']!r} not in {SPLITS}")
        head = str(g["game_id"]).replace("\\", "/").split("/", 1)[0]
        if head != g["split"]:
            raise ValueError(
                f"game {g['probe_id']!r}: game_id starts with {head!r} but split is {g['split']!r}"
            )
        if g["probe_id"] in seen_probe_ids:
            raise ValueError(f"duplicate probe_id {g['probe_id']!r}")
        if g["game_id"] in seen_game_ids:
            raise ValueError(f"duplicate game_id {g['game_id']!r}")
        seen_probe_ids.add(g["probe_id"])
        seen_game_ids.add(g["game_id"])


def resolve_game_file(data_path: str, game_id: str) -> str:
    """Absolute path of the ``game.tw-pddl`` for ``game_id`` under ``data_path``.

    ``data_path`` is the ALFWorld JSON root (the directory that holds the split
    folders, e.g. ``~/.cache/alfworld/json_2.1.1``). Existence is not checked
    here; the env reports a missing game at setup time.
    """
    return os.path.join(os.path.expanduser(data_path), *game_id.split("/"), GAME_FILE_NAME)
