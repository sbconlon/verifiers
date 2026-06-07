"""Phase 4 live spike (runs at the Phase 9 entry gate, needs a GPU + live vLLM).

Go/no-go for the whole action-level pi_hat approach: confirm that scoring a known
token sequence through native /v1/completions (token-id prompt + prompt_logprobs)
reproduces that sequence's *generation* logprobs within fp tolerance, and log
whether requesting prompt_logprobs defeats prefix caching.

This is deliberately standalone (no ALFWorld, no prime-rl) so it can be pulled
forward whenever a GPU serving the SFT checkpoint is free. If it fails with a
systematic (non-fp-noise) divergence that matching temperature/config cannot
reconcile, fall back before building further on pi_hat: a custom /score route
(mirroring /chat/completions/tokens) or in-context continuation tokenization.

Usage:
    python spike_score_consistency.py \
        --base-url http://localhost:8000/v1 --model <served-model> \
        [--prompt "..."] [--tol 5e-3]

Reads:
    OPENAI_API_KEY (defaults to "EMPTY" for a local vLLM).

Exit code 0 if every scored token is within --tol of its generation logprob,
1 otherwise. The per-token max |Δ| and a small table are printed either way.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import httpx
from openai import OpenAI


def _generate(client: OpenAI, model: str, prompt: str, max_tokens: int):
    """Generate a short completion, returning (token_ids, gen_logprobs)."""
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        logprobs=1,
        extra_body=dict(return_tokens_as_token_ids=True),
    )
    choice = resp.choices[0]
    lp = choice.logprobs
    # vLLM with return_tokens_as_token_ids encodes each token as "token_id:N".
    token_ids = [int(t.split("token_id:")[-1]) for t in lp.tokens]
    return token_ids, list(lp.token_logprobs)


def _tokenize(base_url: str, model: str, text: str, api_key: str) -> list[int]:
    """Return the prompt's token ids via vLLM's /tokenize route.

    The route lives at the server root, not under /v1. This gives the exact
    prompt token-ids the scoring call needs as its prefix -- the standalone
    analog of the env's known o_ids at the real gate.

    add_special_tokens=False so the tokenization matches how /completions
    treats the raw prompt string (vLLM uses it as-is); adding specials here
    would shift the prefix and make the spike fail on a tokenization mismatch
    rather than a real scoring divergence.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    resp = httpx.post(
        f"{root}/tokenize",
        json={"model": model, "prompt": text, "add_special_tokens": False},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["tokens"]


def _score(client: OpenAI, model: str, prompt_token_ids: list[int], start: int):
    """Score a known token-id prompt via prompt_logprobs; return per-position
    logprob of the actual token for positions [start, len)."""
    t0 = time.perf_counter()
    resp = client.completions.create(
        model=model,
        prompt=[prompt_token_ids],
        max_tokens=1,
        temperature=1.0,
        top_p=1.0,
        extra_body=dict(prompt_logprobs=1),
    )
    elapsed = time.perf_counter() - t0
    choice = resp.choices[0]
    pl = choice.prompt_logprobs
    out = []
    for p in range(start, len(prompt_token_ids)):
        tok = prompt_token_ids[p]
        entry = pl[p][str(tok)]
        out.append(entry["logprob"] if isinstance(entry, dict) else entry.logprob)
    return out, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--prompt",
        default="<|im_start|>user\nYou are in a kitchen.<|im_end|>\n<|im_start|>assistant\n",
    )
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--tol", type=float, default=5e-3)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    # Prompt token-ids (the scored prefix) + generated token-ids with their
    # generation logprobs. The scored sequence is prompt + generation, and we
    # compare scored vs generation only over the generated positions -- so
    # each scored logprob conditions on the same prefix the generation did.
    prompt_ids = _tokenize(args.base_url, args.model, args.prompt, api_key)
    gen_token_ids, gen_logprobs = _generate(client, args.model, args.prompt, args.max_tokens)

    full_ids = prompt_ids + gen_token_ids
    scored, elapsed = _score(client, args.model, full_ids, start=len(prompt_ids))

    # scored[i] is the logprob of gen_token_ids[i] at full position
    # len(prompt_ids)+i; gen_logprobs[i] is its generation logprob. They must
    # be 1:1 -- a length mismatch means the scoring positions are misaligned
    # (fail loud rather than silently comparing a truncated set).
    if len(scored) != len(gen_logprobs):
        print(f"FAIL - length mismatch: scored={len(scored)} gen={len(gen_logprobs)} "
              "(scoring positions misaligned with generated tokens)")
        return 1
    n = len(scored)
    deltas = [abs(scored[i] - gen_logprobs[i]) for i in range(n)]
    max_delta = max(deltas) if deltas else float("nan")

    print(f"prompt={len(prompt_ids)} tok, generated={len(gen_token_ids)} tok; "
          f"scored {n} generated positions; prompt_logprobs call took {elapsed*1000:.0f} ms")
    print(f"max |Δ logprob| = {max_delta:.3e} (tol {args.tol:.1e})")
    print("idx  token_id  gen_lp      scored_lp   |Δ|")
    for i in range(n):
        print(f"{i:3d}  {gen_token_ids[i]:8d}  {gen_logprobs[i]:+.4f}   {scored[i]:+.4f}   {deltas[i]:.2e}")

    ok = max_delta <= args.tol
    print("PASS" if ok else "FAIL", "- scored logprobs reproduce generation logprobs" if ok else "- systematic divergence; see fallback in the phase doc")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
