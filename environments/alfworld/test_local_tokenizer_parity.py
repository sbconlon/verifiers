"""Parity test: local HF tokenizer vs vLLM /tokenize for ALFWorld message lists.

This test verifies that ALFWorldEnvironment's new _count_tokens implementation
(local HF tokenizer with chat template) produces token counts that match vLLM's
/tokenize endpoint within ±2 tokens for representative ALFWorld message lists.

If the counts diverge by more than 2, the chat template application differs
somewhere (e.g. add_generation_prompt flag, jinja template version mismatch,
or message-content normalization). We need parity within a small tolerance
because the divergence investigation diary's threshold for "fix succeeded" is
gibberish ≤ 5%, and an off-by-N truncation threshold would cause unintended
prompt truncation/non-truncation that could re-introduce gibberish.

Prereqs to run:
  1. vLLM server running at http://localhost:8000 with --model Qwen/Qwen2.5-1.5B-Instruct
  2. transformers installed (already a verifiers dependency)
  3. httpx installed (test-only dep, not used by alfworld_env anymore)

Usage:
  cd /home/ubuntu/verifiers/environments/alfworld
  python3 test_local_tokenizer_parity.py
"""
import asyncio
import json
import sys

import httpx
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
VLLM_TOKENIZE_URL = "http://localhost:8000/tokenize"
TOLERANCE = 2  # max allowed delta between local and remote counts


# Representative ALFWorld message lists at increasing sizes.
# These mirror the structure of real ALFWorld rollouts.
SAMPLE_LISTS = [
    # 1) Just system + first user (turn 0 — what most rollouts saw on Pod #5)
    [
        {"role": "system", "content": "You are an expert agent solving household tasks in a text-based environment. At each step you receive an observation describing your surroundings and a list of admissible actions. Your goal is stated in the first observation."},
        {"role": "user", "content": "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see a armchair 1, a coffeetable 1, a drawer 1, a garbagecan 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 1, a sofa 2, a sofa 1, and a tvstand 1.\n\nYour task is to: examine the tissuebox with the desklamp.\n\nYour admissible actions for this step are:\ngo to armchair 1\ngo to coffeetable 1\ngo to drawer 1\ngo to garbagecan 1\nhelp\ninventory\nlook\n\nNow it's your turn to take an action. You MUST enclose your reasoning within <think> </think> tags, then choose one action from the list above and enclose it within <action> </action> tags."},
    ],
    # 2) Mid-rollout: 5 turns deep (system + 5 user/assistant pairs)
    [
        {"role": "system", "content": "You are an expert agent solving household tasks in a text-based environment."},
        {"role": "user", "content": "Welcome. You see drawer 1, coffeetable 1, sidetable 1. Task: examine tissuebox with desklamp.\n\nActions: go to drawer 1, go to coffeetable 1, look\n\nFollow format."},
        {"role": "assistant", "content": "<think>I should look at the coffeetable for the tissuebox.</think>\n<action>go to coffeetable 1</action>"},
        {"role": "user", "content": "On the coffeetable 1 you see a tissuebox 1.\n\nActions: take tissuebox 1, look, go to drawer 1\n\nFollow format."},
        {"role": "assistant", "content": "<think>Found tissuebox. Now I need a desklamp.</think>\n<action>take tissuebox 1</action>"},
        {"role": "user", "content": "You take the tissuebox 1.\n\nActions: examine tissuebox 1, look, go to sidetable 1\n\nFollow format."},
        {"role": "assistant", "content": "<think>Where is the desklamp? Probably on the sidetable.</think>\n<action>go to sidetable 1</action>"},
        {"role": "user", "content": "On sidetable 1 you see a desklamp 1.\n\nActions: examine tissuebox 1 with desklamp 1, take desklamp 1\n\nFollow format."},
        {"role": "assistant", "content": "<think>Time to examine.</think>\n<action>examine tissuebox 1 with desklamp 1</action>"},
        {"role": "user", "content": "You examine the tissuebox under the desklamp's light. Task complete.\n\nActions: look, inventory\n\nFollow format."},
    ],
    # 3) Long: many turns with verbose content (simulates near-context-limit state)
    [{"role": "system", "content": "You are an expert agent."}] + [
        {"role": "user", "content": "Turn " + str(i) + ": " + ("Long description of room contents and many admissible actions. " * 8)}
        if i % 2 == 0 else
        {"role": "assistant", "content": "<think>Reasoning step " + str(i) + "</think>\n<action>look</action>"}
        for i in range(20)
    ],
]


async def vllm_count(client: httpx.AsyncClient, messages: list) -> int:
    """Get vLLM's /tokenize count for a message list."""
    resp = await client.post(
        VLLM_TOKENIZE_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "add_generation_prompt": True,
        },
    )
    resp.raise_for_status()
    return resp.json()["count"]


def local_count(tokenizer, messages: list) -> int:
    """Get local HF tokenizer count for a message list."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


async def main():
    print(f"Loading local tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  ok\n")

    print(f"Hitting vLLM /tokenize at {VLLM_TOKENIZE_URL}")
    async with httpx.AsyncClient(timeout=10.0) as http:
        # Sanity: confirm server up
        try:
            r = await http.get("http://localhost:8000/v1/models")
            r.raise_for_status()
        except Exception as e:
            print(f"  ERROR: vLLM not reachable: {e}")
            print(f"  Start vLLM first: vllm serve {MODEL_NAME} --port 8000")
            return 2

        print(f"  ok\n")

        all_pass = True
        for i, messages in enumerate(SAMPLE_LISTS):
            local = local_count(tokenizer, messages)
            remote = await vllm_count(http, messages)
            delta = local - remote
            status = "PASS" if abs(delta) <= TOLERANCE else "FAIL"
            if status == "FAIL":
                all_pass = False
            n_msgs = len(messages)
            print(
                f"Sample {i+1} ({n_msgs:>2} messages): "
                f"local={local:>5}  remote={remote:>5}  "
                f"delta={delta:+d}  [{status}]"
            )

    print()
    if all_pass:
        print(f"OVERALL: PASS — all samples within ±{TOLERANCE} tokens")
        return 0
    else:
        print(f"OVERALL: FAIL — at least one sample exceeded ±{TOLERANCE} tokens")
        print("Likely cause: chat template difference between local transformers and vLLM's bundled tokenizer.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
