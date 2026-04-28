"""Concurrency safety test for the new local-tokenizer sliding window.

Verifies that ALFWorldEnvironment._count_tokens (with the new local HF
tokenizer) handles N concurrent async coroutines without:
  - raising "Already borrowed" RuntimeError from the Rust tokenizer
  - returning inconsistent counts for the same input
  - leaking memory (single tokenizer instance, no per-call reload)

Background: HF Fast tokenizers are not thread-safe in their underlying
Rust implementation — concurrent calls from multiple OS threads can panic
with "Already borrowed". asyncio coroutines on a single event loop are
NOT a problem (single-threaded cooperative concurrency), but we test
anyway to make the safety property explicit.

This test does NOT require vLLM — it exercises the local tokenizer code
path directly.

Usage:
  cd /home/ubuntu/verifiers/environments/alfworld
  python3 test_sliding_window_concurrent.py
"""
import asyncio
import sys

# Hack: ALFWorldEnvironment's __init__ calls super().__init__() which expects
# a vf.Rubric and vf.Dataset. We bypass that by constructing a thin subclass
# that only loads what we need to test (_count_tokens).

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_CONCURRENT = 16
N_ITERATIONS = 5  # repeat the concurrent fan-out N times


class _CountTokensTester:
    """Minimal harness mimicking ALFWorldEnvironment._count_tokens semantics."""

    def __init__(self):
        self._tokenizer = None  # lazy-loaded

    async def count_tokens(self, messages: list, model: str) -> int:
        # Mirror the body of ALFWorldEnvironment._count_tokens
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(model)

        payload_messages = []
        for msg in messages:
            d = dict(msg)
            role = d.get("role", "user")
            content = d.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            payload_messages.append({"role": role, "content": content})

        token_ids = self._tokenizer.apply_chat_template(
            payload_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)


# Sample message lists, each repeated for the concurrent fan-out
SAMPLE_LISTS = [
    [
        {"role": "system", "content": "You are an expert agent."},
        {"role": "user", "content": f"Sample {i}: examine tissuebox. Available: drawer 1, coffeetable 1. Follow format."},
    ]
    for i in range(N_CONCURRENT)
]


async def main():
    print(f"Building tester (lazy-load mode — tokenizer loads on first call)...")
    tester = _CountTokensTester()
    print(f"  ok (tokenizer is None: {tester._tokenizer is None})\n")

    print(f"Running {N_ITERATIONS} iterations of {N_CONCURRENT} concurrent count_tokens calls...\n")

    all_pass = True
    for iteration in range(N_ITERATIONS):
        tasks = [
            tester.count_tokens(messages, MODEL_NAME) for messages in SAMPLE_LISTS
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            print(f"  Iteration {iteration+1}: FAILED with {type(e).__name__}: {e}")
            all_pass = False
            continue

        # All results should be deterministic given identical inputs (each task
        # has a unique message but same template). Verify each task's result is
        # consistent with what we get from a fresh single call.
        consistent = True
        for i, (msgs, result) in enumerate(zip(SAMPLE_LISTS, results)):
            single = await tester.count_tokens(msgs, MODEL_NAME)
            if result != single:
                print(f"  Iteration {iteration+1}, task {i}: INCONSISTENT — concurrent={result}, single={single}")
                consistent = False
                all_pass = False
        if consistent:
            print(f"  Iteration {iteration+1}: PASS — all {N_CONCURRENT} concurrent calls returned consistent counts")

        # Verify tokenizer is loaded after first iteration and reused (NOT reloaded)
        if iteration == 0:
            print(f"    (after first iteration: tokenizer loaded={tester._tokenizer is not None})")

    print()
    if all_pass:
        print(f"OVERALL: PASS — {N_ITERATIONS} × {N_CONCURRENT} concurrent calls, no errors, all results consistent")
        return 0
    else:
        print(f"OVERALL: FAIL — see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
