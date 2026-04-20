"""Light tests for Step 2 prompting changes.

Checks:
  - SYSTEM_PROMPT contains role context and no format instructions
  - _format_obs() uses new admissible actions framing and appends FORMAT_REMINDER
  - FORMAT_REMINDER contains both <think> and <action> tag instructions
  - build_dataset() embeds the updated SYSTEM_PROMPT in each row

Run with:
    cd ~/verifiers && ~/prime-rl/.venv/bin/python environments/alfworld/test_prompting.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock verifiers before importing alfworld_env to avoid broken experimental
# sandbox import in the prime-rl venv (prime_sandboxes version mismatch).
_vf_mock = MagicMock()
_vf_mock.MultiTurnEnv = object  # ALFWorldEnvironment inherits from this
sys.modules["verifiers"] = _vf_mock

# Ensure the verifiers repo root is on the path.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environments.alfworld.alfworld_env import (  # noqa: E402
    FORMAT_REMINDER,
    SYSTEM_PROMPT,
    ALFWorldEnvironment,
)


def test_system_prompt_is_role_only():
    assert "expert agent" in SYSTEM_PROMPT, "SYSTEM_PROMPT should describe the agent role"
    assert "admissible actions" in SYSTEM_PROMPT, "SYSTEM_PROMPT should mention admissible actions"
    assert "<think>" not in SYSTEM_PROMPT, "Format instructions must not appear in SYSTEM_PROMPT"
    assert "<action>" not in SYSTEM_PROMPT, "Format instructions must not appear in SYSTEM_PROMPT"
    assert "exactly this format" not in SYSTEM_PROMPT, "Old format instruction text found in SYSTEM_PROMPT"
    print("PASS test_system_prompt_is_role_only")


def test_format_reminder_contains_tag_instructions():
    assert "<think>" in FORMAT_REMINDER
    assert "</think>" in FORMAT_REMINDER
    assert "<action>" in FORMAT_REMINDER
    assert "</action>" in FORMAT_REMINDER
    assert "MUST" in FORMAT_REMINDER, "FORMAT_REMINDER should be emphatic about the requirement"
    print("PASS test_format_reminder_contains_tag_instructions")


def test_format_obs_structure():
    obs = "You are in a kitchen. You see a knife."
    commands = ["go north", "take knife", "look"]
    result = ALFWorldEnvironment._format_obs(obs, commands)

    assert obs in result, "Observation text missing from formatted output"
    assert "Your admissible actions for this step are:" in result, "New actions framing missing"
    assert "Available actions:" not in result, "Old actions framing still present — not fully replaced"
    for cmd in commands:
        assert cmd in result, f"Command '{cmd}' missing from formatted output"
    assert FORMAT_REMINDER in result, "FORMAT_REMINDER not present in formatted output"
    assert result.endswith(FORMAT_REMINDER), "FORMAT_REMINDER must be the last thing in the message"
    print("PASS test_format_obs_structure")


def test_format_obs_reminder_is_proximate_to_actions():
    """FORMAT_REMINDER must come immediately after the last action, with no intervening text."""
    obs = "You see a table."
    commands = ["go east", "look"]
    result = ALFWorldEnvironment._format_obs(obs, commands)

    last_command_end = result.rindex("look") + len("look")
    reminder_start = result.index(FORMAT_REMINDER)
    assert reminder_start == last_command_end, (
        f"FORMAT_REMINDER does not immediately follow the action list.\n"
        f"Text between last action and reminder: {result[last_command_end:reminder_start]!r}"
    )
    print("PASS test_format_obs_reminder_is_proximate_to_actions")


def test_build_dataset_uses_updated_system_prompt(tmp_path: Path):
    from environments.alfworld.alfworld_env import build_dataset

    game_dir = tmp_path / "valid_seen" / "task_001"
    game_dir.mkdir(parents=True)
    (game_dir / "game.tw-pddl").write_text("")

    ds = build_dataset(str(tmp_path), "valid_seen")
    assert len(ds) == 1
    system_msg = ds[0]["prompt"][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == SYSTEM_PROMPT, (
        "build_dataset row does not carry the updated SYSTEM_PROMPT"
    )
    assert "<think>" not in system_msg["content"], "Old format instructions found in dataset system prompt"
    print("PASS test_build_dataset_uses_updated_system_prompt")


if __name__ == "__main__":
    import tempfile
    test_system_prompt_is_role_only()
    test_format_reminder_contains_tag_instructions()
    test_format_obs_structure()
    test_format_obs_reminder_is_proximate_to_actions()
    with tempfile.TemporaryDirectory() as tmp:
        test_build_dataset_uses_updated_system_prompt(Path(tmp))
    print("\nAll tests passed.")
