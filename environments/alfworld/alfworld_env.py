"""ALFWorld environment for prime-rl / verifiers.

Environment contract:
  - setup_state():   load game from state["info"]["game_file"], reset, inject initial obs
  - env_response():  parse <action> from last AssistantMessage, step game, return UserMessage
  - cleanup:         remove non-serializable tw_env from state before ZMQ serialization
  - reward:          1.0 if state["won"] else 0.0 (set by env_response on terminal step)

Context window truncation (optional, enabled when max_context_tokens > 0):
  - get_prompt_messages() enforces the token budget before each LLM call.
  - Eviction policy: FIFO from messages[1:] — messages[0] (system prompt) is always kept.
    The initial task observation (messages[1]) is evictable: losing it creates strong partial
    observability and maximally satisfies the strategic relevance test (two trajectories with
    identical recent context but different evicted tasks have widely divergent optimal actions).
  - Token counting: a locally-loaded HuggingFace tokenizer is applied with the model's chat
    template (lazy-loaded on first use, one tokenizer per env worker process). Local
    tokenization replaces an earlier design that POSTed to vLLM's /tokenize endpoint per turn;
    that earlier design saturated vLLM's tokenizer thread pool under 64 concurrent multi-turn
    rollouts and indirectly corrupted concurrent generation output (the eval-vs-training
    divergence investigation, see diary 20260427-eval-training-divergence-investigation §3).
  - Truncation metrics written to state for §5 reporting:
      state["context_truncated"]          bool  — whether any eviction occurred this episode
      state["context_truncated_at_turn"]  int   — trajectory step index of first eviction
      state["context_evictions"]          int   — total messages evicted across the episode

Training-time probe capture (optional, probe_capture=True; training-probe plan, Phase 1):
  - get_model_response executes the greedy response and draws m sampled reasoning
    blocks at the training temperature for the pi_hat estimate (D-4).
  - add_trajectory_step writes the full conditional matrix, the context-window state
    and the oracle proposal into extras["probe"] (plan/training-probe/probe-schema.md §1).
  - oracle="handcoded" runs alfworld's HandCodedTWAgent in-env (D-6); "planner" reads
    the engine's policy_commands.
  With probe_capture unset and oracle="off" every path above is byte-identical to before.
"""
import asyncio
import contextlib
import datetime
import logging
import os
import random
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import verifiers as vf
from datasets import Dataset
from openai import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EPISODE_STEPS = 50

SYSTEM_PROMPT = (
    "You are an expert agent solving household tasks in a text-based environment. "
    "At each step you receive an observation describing your surroundings and a list "
    "of admissible actions. Your goal is stated in the first observation."
)

FORMAT_REMINDER = (
    "\n\nNow it's your turn. Respond with EXACTLY this format:\n"
    "<think>your reasoning here</think>\n"
    "<action>your chosen action here</action>\n\n"
    "Use the literal angle-bracket tags shown above. "
    "Do NOT use square brackets -- the format must be "
    "<action>command</action>, NOT [action]command[/action]."
)

# Training-time probe (Phase 1).
ORACLE_MODES = ("off", "handcoded", "planner")
PROBE_SCHEMA_VERSION = 1
# Oracle record when no oracle runs (probe-schema.md §1 "oracle"); copied per turn.
_ORACLE_OFF: dict[str, Any] = {"action": None, "status": "off", "plan": None, "steps_left": None}
_ORACLE_FIELDS = tuple(_ORACLE_OFF)

# Per-process cache: game_file -> tw_env_id registered with textworld.gym.
# Populated lazily on first rollout of each game file; lock only needed on
# cache miss (i.e. first time a given game file is seen in this process).
_TW_ENV_ID_CACHE: dict[str, str] = {}
_TW_REGISTER_LOCK = threading.Lock()
_TW_PARSE_LOCK = threading.Lock()


@contextlib.contextmanager
def _logged_parse_lock(phase: str, game_file: str):
    """Instrumented acquire/hold/release of the process-global _TW_PARSE_LOCK.

    Emits INFO lines so wedge diagnosis is disciplined instead of guessed. These
    land in the per-worker ``env_worker_<id>.log`` ONLY when the process that
    spawns the env server passes ``log_dir`` (the RL orchestrator and the
    warm-start collect both must, or worker logs go nowhere -- see
    vf_utils.spawn_env_server, which forces console_logging=False).

    Reading the log:
      * ``HOLDING ... phase=P game=G`` with NO matching ``released`` line for the
        same (game,pid) => that rollout is WEDGED inside phase P (make | reset |
        step) holding the lock. G + pid is the culprit; py-spy the pid.
      * ``waiting ... game=G'`` with no later ``HOLDING`` for G' => G' is a VICTIM
        blocked behind the culprit, not itself broken.
      * ``held=<t>s`` on the ``released`` line distinguishes a SLOW grounding
        (large but finite t) from an INFINITE wedge (no released line at all).
    """
    pid = os.getpid()
    logger.info(f"[lock] waiting  phase={phase} game={game_file} pid={pid}")
    t_wait = time.monotonic()
    with _TW_PARSE_LOCK:
        t_held = time.monotonic()
        logger.info(
            f"[lock] HOLDING  phase={phase} game={game_file} pid={pid} "
            f"waited={t_held - t_wait:.2f}s"
        )
        try:
            yield
        finally:
            logger.info(
                f"[lock] released phase={phase} game={game_file} pid={pid} "
                f"held={time.monotonic() - t_held:.2f}s"
            )


def _make_demangler(env):
    """Factory: fresh AlfredDemangler(shuffle=False) per gym.make() call.

    AlfredDemangler must NOT be passed as a shared instance to register_games()
    because textworld._make_env() calls wrapper(env) which mutates _wrapped_env
    on the same object — causing rollouts to corrupt each other's wrapper chain.
    """
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    d = AlfredDemangler(shuffle=False)
    d._wrapped_env = env
    return d


# ---------------------------------------------------------------------------
# Action-level ARM: teacher-forcing scorer (Phase 4)
# ---------------------------------------------------------------------------
# The pi_hat scorer lives in this (editable) env module rather than in the
# verifiers core client: it is action-level-ARM-only, its sole caller is this
# env, and co-locating it here keeps the change in one live-editable place
# (consistent with Phase 2's "co-locate math with its caller"). It uses the
# token client's public token_client.post to hit native /v1/completions with
# prompt_logprobs -- no custom server route (custom_build_app only *adds* routes,
# so the native endpoint survives the prime-rl wrapper).


# Permissive response model for /v1/completions with prompt_logprobs. Each choice
# carries prompt_logprobs: a list (indexed by prompt token position) where each
# entry is None (first position) or a dict keyed by str(token_id) ->
# {"logprob": float, "rank": int, "decoded_token": str}.
class CompletionScoreChoice(BaseModel):
    index: int
    prompt_logprobs: Optional[list[Optional[dict[str, Any]]]] = None


class CompletionWithPromptLogprobs(BaseModel):
    choices: list[CompletionScoreChoice]


# Response of the custom /v1/score route: one summed span logprob per prompt.
class ScoreResponse(BaseModel):
    scores: list[float]


def _build_score_body(prompts_token_ids: list[list[int]], model: str, temperature: float) -> dict:
    """Build the /v1/completions body for pure scoring (no generation).

    Neutral config (T=1, top_p=1, no penalties/bias) so prompt_logprobs are the
    *raw* policy distribution -- pi_hat is renorm(log_softmax(z)) over A(o), and
    T=1 is the identity that yields raw in either vLLM prompt_logprobs regime.
    max_tokens=1 because a pure scoring call still forwards the prompt; we read
    prompt_logprobs, not the generated token.
    """
    return dict(
        model=model,
        prompt=prompts_token_ids,  # vLLM accepts list[list[int]]
        max_tokens=1,
        temperature=temperature,
        top_p=1.0,
        # prompt_logprobs must be a TOP-LEVEL field: this body is sent via a raw
        # client.token_client.post(body=...), which (unlike client.completions.create)
        # does NOT merge an "extra_body" key into the request. Nesting it under
        # extra_body made vLLM silently ignore it -> prompt_logprobs=None in the
        # response -> NoneType subscript in _span_logprob_sum.
        prompt_logprobs=1,
    )


def _span_logprob_sum(choice_prompt_logprobs: list, prompt_token_ids: list[int], start: int, end: int) -> float:
    """Sum actual-token logprobs over [start, end), read BY TOKEN-ID (not by rank
    -- rank 0 is not guaranteed to be the actual token). Each prompt_logprobs[p]
    is a dict {str(token_id): entry}; entry is a Mapping with a "logprob" key
    (JSON-deserialized vLLM Logprob) or an object with a .logprob attribute."""
    total = 0.0
    for p in range(start, end):
        tok = prompt_token_ids[p]
        entry = choice_prompt_logprobs[p][str(tok)]
        total += entry["logprob"] if isinstance(entry, Mapping) else entry.logprob
    return total


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ALFWorldEnvironment(vf.MultiTurnEnv):

    def __init__(
        self,
        max_context_tokens: int = -1,
        log_trajectories: str = "none",
        tokenizer_path: str | None = None,
        num_reasoning_blocks: int = 1,
        use_nm_fusion: bool = True,
        prewarm_pi_hat_prefixes: bool = True,
        use_score_route: bool = True,
        probe_capture: bool = False,
        probe_block_temperature: float = 0.8,
        probe_block_min_p: float = 0.1,
        oracle: str = "off",
        store_raw_logps: bool = False,
        pi_hat_renorm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_context_tokens = max_context_tokens
        self.log_trajectories = log_trajectories  # "none" | "wins" | "all"
        # Action-level ARM (Phase 3): number of reasoning blocks m sampled per
        # decision point for the RBMC pi_hat estimate. 1 == standard single
        # generation (GRPO/PPO and any unconfigured run are then byte-for-byte
        # unchanged). m=4 for action-level ARM (D-5).
        self.num_reasoning_blocks = num_reasoning_blocks
        # Phase 8/9 perf: generate the m reasoning blocks with a single n=m request
        # (vLLM prefills o once, samples m off it) instead of m separate requests
        # that each re-prefill the long ALFWorld context. Critical for rollout
        # throughput; falls back to m separate generations if the inference server
        # doesn't return m token-complete choices.
        self.use_nm_fusion = use_nm_fusion
        # Phase 9 perf: pre-warm each [o, R_j] prefix into the prefix cache before the
        # batched pi_hat scoring, so each action prompt reuses the ~2000-token prefix
        # instead of recomputing it (a single batched request prefills its prompts
        # simultaneously, which defeats in-batch prefix reuse). Set False to A/B it.
        self.prewarm_pi_hat_prefixes = prewarm_pi_hat_prefixes
        # Phase 9 perf: use the custom /v1/score route (server-side span summation,
        # one float per prompt) instead of pulling full per-position prompt_logprobs
        # over /v1/completions. Falls back to /v1/completions if the route is absent.
        self.use_score_route = use_score_route
        # Explicit path/HF-id for the tokenizer used by _count_tokens. When set,
        # overrides state["model"] (which is unreliable in LoRA + checkpoint-resume
        # setups: vLLM advertises the adapter NAME, not a real path, so a
        # naive AutoTokenizer.from_pretrained(state["model"]) fails). Set this
        # to the base model path in load_environment kwargs whenever using LoRA.
        # See diary 20260502 grpo-1.5b-sft-run-001 for the live diagnosis.
        self._tokenizer_path = tokenizer_path
        # Local HF tokenizer for context-token counting. Lazy-loaded on first
        # _count_tokens() call (typically from an EnvWorker process); the
        # orchestrator process also instantiates ALFWorldEnvironment for buffer
        # construction but never calls _count_tokens, so the tokenizer is never
        # loaded there — saving ~200 MB resident in the orchestrator process.
        self._tokenizer = None
        # Training-time probe (training-probe plan, Phase 1). probe_capture=True
        # switches get_model_response to "greedy executed response + m sampled
        # reasoning blocks at the training temperature" (D-4) and makes
        # add_trajectory_step write the full pi_hat matrix, the window state and
        # the oracle proposal into extras["probe"] (probe-schema.md §1). Off by
        # default; with it unset the training path is byte-identical (D-12).
        if oracle not in ORACLE_MODES:
            raise ValueError(f"oracle must be one of {ORACLE_MODES}, got {oracle!r}")
        if probe_capture and num_reasoning_blocks < 2:
            raise ValueError(
                "probe_capture requires num_reasoning_blocks >= 2 (the pi_hat matrix "
                f"needs sampled blocks); got {num_reasoning_blocks}"
            )
        self.probe_capture = probe_capture
        self.probe_block_temperature = probe_block_temperature
        self.probe_block_min_p = probe_block_min_p
        self.oracle = oracle
        self.store_raw_logps = store_raw_logps
        # pi_hat convention (training-probe D-13, 2026-11-07). True: each block's
        # conditionals are the softmax over the admissible union (rows sum to 1,
        # the convention every run before warm-003 used). False: the raw sequence
        # probabilities exp(summed logp) with no renormalization, so pi_hat(a*) is
        # the probability of emitting the action string itself and the per-block
        # admissible mass exp(Z_j) <= 1 is left in the record for inspection.
        # Applies to BOTH the training path (_compute_pi_hat) and probe capture.
        self.pi_hat_renorm = pi_hat_renorm

    # ------------------------------------------------------------------
    # TextWorld helpers
    # ------------------------------------------------------------------

    def _get_tw_env_id(self, game_file: str) -> str:
        """Return the textworld gym id for game_file, registering it if needed.

        Registration is done at most once per game file per process (lazy cache).
        The lock is only contended on the first rollout of each game file.
        """
        # Probe oracle modes request extra EnvInfos (facts for the handcoded
        # expert, policy_commands for the planner), so they register under their
        # own cache key. The training path's key and request_infos are unchanged.
        oracle = getattr(self, "oracle", "off")
        cache_key = game_file if oracle == "off" else f"{game_file}::oracle={oracle}"
        if cache_key in _TW_ENV_ID_CACHE:
            return _TW_ENV_ID_CACHE[cache_key]

        with _TW_REGISTER_LOCK:
            # Re-check inside the lock: another thread may have registered while
            # we were waiting.
            if cache_key not in _TW_ENV_ID_CACHE:
                import textworld
                import textworld.gym
                from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

                info_kwargs: dict[str, Any] = dict(won=True, admissible_commands=True, extras=["gamefile"])
                if oracle == "handcoded":
                    info_kwargs["facts"] = True  # BasePolicy.get_facts reads game_state["facts"]
                elif oracle == "planner":
                    info_kwargs["policy_commands"] = True  # fast_downward plan per step
                request_infos = textworld.EnvInfos(**info_kwargs)
                tw_env_id = textworld.gym.register_games(
                    [game_file],
                    request_infos,
                    batch_size=1,
                    asynchronous=False,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    wrappers=[_make_demangler, AlfredInfos],
                )
                _TW_ENV_ID_CACHE[cache_key] = tw_env_id

        return _TW_ENV_ID_CACHE[cache_key]

    def _make_tw_env(self, game_file: str):
        """Instantiate and return a fresh TextWorld gym env for one game file.

        gym.make() constructs the wrapper chain and may touch the tatsu textgen
        parser during init. Serialise with _TW_PARSE_LOCK to avoid concurrent
        threads corrupting the module-level _PARSER state.
        """
        import textworld.gym
        tw_env_id = self._get_tw_env_id(game_file)
        with _logged_parse_lock("make", game_file):
            return textworld.gym.make(tw_env_id)

    @staticmethod
    def _format_obs(obs: str, admissible_commands: list[str]) -> str:
        """Combine observation text with the list of available actions and format reminder."""
        commands = "\n".join(admissible_commands)
        return (
            f"{obs}\n\n"
            f"Your admissible actions for this step are:\n{commands}"
            f"{FORMAT_REMINDER}"
        )

    # ------------------------------------------------------------------
    # Context window truncation
    # ------------------------------------------------------------------

    def _ensure_tokenizer(self, state: vf.State) -> None:
        """Lazily load the local HF tokenizer (shared by _count_tokens and the
        action-level pi_hat continuation tokenization). ~200 MB, one-time per env
        worker process. Source resolution: self._tokenizer_path (set this to the
        base model path for LoRA), else state["model"]."""
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer
        tokenizer_source = self._tokenizer_path or state["model"]
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except OSError as e:
            # Most common cause: LoRA setup where state["model"] is the adapter
            # name (no on-disk tokenizer). Re-raise with a hint so the caller's
            # try/except logs an actionable message.
            raise OSError(
                f"Could not load tokenizer from {tokenizer_source!r}. "
                f"If using LoRA, set tokenizer_path explicitly in "
                f"load_environment kwargs to the base model path "
                f"(self._tokenizer_path={self._tokenizer_path!r}, "
                f"state['model']={state.get('model')!r}). "
                f"Original error: {e}"
            ) from e
        logger.info(
            f"Loaded local tokenizer for {tokenizer_source} "
            f"(env worker pid={os.getpid()})"
        )

    async def _count_tokens(self, messages: vf.Messages, state: vf.State) -> int:
        """Return the exact token count for messages using a local HF tokenizer.

        Lazily loads the HuggingFace tokenizer on first call (~200 MB, one-time
        cost per env worker process; reused across all rollouts on this worker).
        Applies the model's chat template with add_generation_prompt=True to
        match what vLLM does server-side before inference — i.e. the count
        returned here equals what vLLM will see when it receives the same
        message list.

        Tokenizer source resolution (in priority order):
          1. self._tokenizer_path (set via load_environment(tokenizer_path=...))
          2. state["model"] — the served-model identifier from prime-rl

        For LoRA setups, ALWAYS set tokenizer_path explicitly to the base model
        path. state["model"] is unreliable there: when prime-rl resumes from a
        checkpoint, vLLM serves the LoRA adapter under its `name` (e.g.
        "grpo-1.5b-sft-run-001") which is not a real filesystem path and not
        an HF Hub repo, so AutoTokenizer.from_pretrained on it raises OSError.
        Tokenization is invariant under LoRA — adapters do not change the
        tokenizer — so always use the base model path. See diary
        20260502 grpo-1.5b-sft-run-001 (post-resume divergence triage).

        This replaces an earlier implementation that POSTed to vLLM's /tokenize
        endpoint. Under 64 concurrent multi-turn rollouts that earlier design
        saturated vLLM's tokenizer thread pool and indirectly corrupted
        concurrent generation output (see diary
        20260427-eval-training-divergence-investigation §3). Local tokenization
        eliminates the /tokenize traffic entirely.

        Concurrency: this method is called from async coroutines on the env
        worker's single asyncio event loop. The tokenizer call below is sync;
        asyncio guarantees no concurrent execution within this coroutine, so
        no locking is required despite HF Fast tokenizers being thread-unsafe
        in the underlying Rust implementation.
        """
        self._ensure_tokenizer(state)

        # Serialise messages to plain role/content dicts (matches what HF chat
        # template expects).
        payload_messages = []
        for msg in messages:
            d = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
            role = d.get("role", "user")
            content = d.get("content") or ""
            # Flatten list content parts to a single string (ALFWorld is text-only).
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            payload_messages.append({"role": role, "content": content})

        # Apply chat template, then tokenize as an explicit two-step. We do NOT
        # use apply_chat_template(tokenize=True) directly: some transformers
        # versions return a BatchEncoding dict (input_ids + attention_mask)
        # rather than a flat token list, and len() of that dict is 2 — silently
        # corrupting our token counts. The string→encode path is robust.
        # add_generation_prompt=True matches what vLLM does before inference.
        # add_special_tokens=False avoids double-adding BOS/EOS that the chat
        # template already inserts (<|im_start|>...<|im_end|>).
        text = self._tokenizer.apply_chat_template(
            payload_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids)

    async def _apply_sliding_window(
        self, messages: vf.Messages, state: vf.State
    ) -> vf.Messages:
        """Evict oldest messages until the token count fits within budget.

        messages[0] (system prompt) is always preserved.
        messages[1:] are eviction candidates, removed FIFO (oldest first).

        After each eviction the token count is re-computed via _count_tokens
        (local HF tokenizer; see its docstring for concurrency / lazy-load notes).
        In practice ALFWorld episodes evict at most 1-2 messages per turn, so
        the loop runs at most 2-3 iterations.

        Truncation metrics are recorded in state on first eviction:
          state["context_truncated"]          bool
          state["context_truncated_at_turn"]  int   (trajectory step index)
          state["context_evictions"]          int   (cumulative across episode)
        """
        count = await self._count_tokens(messages, state)
        if count <= self.max_context_tokens:
            return messages

        messages = list(messages)  # mutable copy; messages[0] is system prompt

        while count > self.max_context_tokens:
            if len(messages) <= 1:
                # Only the system prompt remains — cannot evict further.
                logger.warning(
                    f"Context budget ({self.max_context_tokens} tokens) is smaller "
                    f"than the system prompt alone ({count} tokens). "
                    f"Returning system prompt only."
                )
                break

            evicted = messages.pop(1)
            evicted_role = getattr(evicted, "role", "?")
            evicted_chars = len(str(getattr(evicted, "content", "") or ""))
            # Evict the paired assistant response together with its user prompt
            # so the context never contains an orphaned reply without its question.
            if (
                evicted_role == "user"
                and len(messages) > 1
                and getattr(messages[1], "role", None) == "assistant"
            ):
                paired = messages.pop(1)
                evicted_chars += len(str(getattr(paired, "content", "") or ""))
                state["context_evictions"] = state.get("context_evictions", 0) + 1
            logger.debug(
                f"Evicted {evicted_role} message (~{evicted_chars} chars) "
                f"at turn {len(state['trajectory'])}. "
                f"Remaining messages: {len(messages)}."
            )

            # Record truncation metrics on first eviction in this episode.
            if not state.get("context_truncated"):
                state["context_truncated"] = True
                state["context_truncated_at_turn"] = len(state["trajectory"])
                state["context_evictions"] = 0
            state["context_evictions"] += 1

            count = await self._count_tokens(messages, state)

        return messages

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Build the prompt for the next LLM call, applying context truncation if configured.

        Calls the parent implementation first (which appends the latest env response),
        then enforces max_context_tokens via _apply_sliding_window if set.
        Truncation errors are caught and logged — on failure the untruncated messages
        are returned so the rollout can continue (vLLM will reject if truly too long).
        """
        messages = await super().get_prompt_messages(state)
        if self.max_context_tokens > 0:
            try:
                messages = await self._apply_sliding_window(messages, state)
            except Exception as exc:
                logger.warning(
                    f"Token counting failed ({type(exc).__name__}: {exc!r}); skipping truncation this turn. "
                    f"vLLM will reject if prompt exceeds its context limit."
                )
        return messages

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    async def get_model_response(self, state: vf.State, prompt: vf.Messages, *args, **kwargs):
        """Action-level ARM (Phase 3): sample m reasoning blocks per decision point.

        At m == 1 this is a literal passthrough to super().get_model_response, so
        GRPO/PPO and any unconfigured run are unchanged. At m > 1 it issues m full
        generations against the *same* prompt (the shared observation o, so vLLM
        prefix-caches the o prefill and only the m decodes are new work), picks j*
        uniformly (seeded for reproducibility), stashes all m responses + j* in
        transient state, and returns the executed (j*-th) response.

        A full generation samples (R_j, a_j) jointly and autoregressively
        a_j ~ π(·|o, R_j); executing responses[j*] realizes the diary's "sample
        R_{j*}, then a* ~ π(·|o, R_{j*})". The m-1 non-executed blocks' reasoning
        is what Phase 4 teacher-forces to estimate the marginal π̂.

        Stash contract consumed by Phase 4 (add_trajectory_step), overwritten each
        turn, popped at rollout end (cleanup_alf_env) so it never serializes:
            state["_pending_reasoning_blocks"]: list[Response]  # length m
            state["_executed_block_idx"]:       int             # j*
        """
        m = self.num_reasoning_blocks
        # super() must be resolved in method scope (zero-arg super() does not work
        # inside the comprehension below); the bound method is then called m times.
        parent_get_response = super().get_model_response
        if getattr(self, "probe_capture", False):
            # Training-time probe (Phase 1, D-4): greedy executed response plus m
            # sampled blocks for the pi_hat matrix. Never taken on the training path.
            return await self._probe_get_model_response(
                state, prompt, parent_get_response, m, *args, **kwargs
            )
        if m == 1:
            return await parent_get_response(state, prompt, *args, **kwargs)

        # Phase 8/9 perf (CRITICAL): generate the m reasoning blocks with ONE n=m
        # request so vLLM prefills the (long ~5-9k-token) observation o ONCE and
        # samples m sequences off it in a single batched decode. The fallback below
        # -- m separate same-prompt requests -- relies on auto prefix caching, but
        # concurrent identical requests do NOT share the prefill until one completes,
        # so each re-prefills o: ~m x the prefill per turn, which dominated the
        # >10 min/rollout seen at Phase 9. Fusion eliminates that.
        responses = None
        if getattr(self, "use_nm_fusion", True):
            try:
                responses = await self._generate_m_blocks_fused(state, prompt, m)
            except Exception as exc:  # noqa: BLE001 -- never fail the rollout on fusion
                logger.warning(
                    f"n=m generation fusion failed ({exc!r}); falling back to m "
                    "separate generations (slow). Check the inference server's n>1 "
                    "support on /v1/chat/completions/tokens."
                )
                responses = None

        if responses is None:
            # Fallback: m separate same-prompt generations (auto-prefix-cache only;
            # correct but slow on long contexts -- see above).
            responses = list(
                await asyncio.gather(
                    *[parent_get_response(state, prompt, *args, **kwargs) for _ in range(m)]
                )
            )

        # Uniform j*, seeded from (trajectory_id, turn) for reproducibility. A str
        # seed gives random.Random a stable (sha512-derived) state -- deterministic
        # across processes, unlike the hash-randomized built-in hash() on strings.
        # len(trajectory) is this turn's index (the step is appended afterwards).
        rng = random.Random(f"{state['trajectory_id']}:{len(state['trajectory'])}")
        j_star = rng.randrange(len(responses))

        state["_pending_reasoning_blocks"] = list(responses)
        state["_executed_block_idx"] = j_star
        return responses[j_star]

    async def _generate_m_blocks_fused(
        self, state: vf.State, prompt, m: int, sampling_args: dict | None = None
    ):
        """Generate the m reasoning blocks with a SINGLE n=m request (Phase 8/9).

        Mirrors verifiers' Client.get_response but (a) sets sampling n=m so the
        inference server prefills o once and samples m sequences in one batched
        decode, and (b) slices the m returned choices into m vf.Response objects via
        the client's own from_native_response parser -- so token_ids / logprobs are
        extracted exactly as for a single generation, and the m blocks are i.i.d.
        samples from pi(.|o) just like the m-separate path.

        Returns the m responses, or None to signal the caller to fall back (server
        returned < m choices, or a choice lacked per-token data -- meaning the
        custom /tokens route didn't emit token ids for n>1, in which case the
        m-separate fallback is the correct path).
        """
        client = state["client"]
        model = state["model"]
        # sampling_args: optional override of the state's args (the probe passes the
        # block temperature / min_p here); default is today's behavior.
        base_args = sampling_args if sampling_args is not None else (state.get("sampling_args") or {})
        sampling_args = {**base_args, "n": m}
        tools = state.get("tool_defs")

        # Replicate Client.get_response's plumbing (headers + native conversion),
        # but with n=m and multi-choice slicing.
        call_kwargs = {"state": state}
        headers = client._build_state_headers(state)
        if headers:
            call_kwargs["extra_headers"] = headers

        native_prompt, extra_kwargs = await client.to_native_prompt(prompt)
        native_tools = await client.to_native_tools(tools)
        native = await client.get_native_response(
            native_prompt, model, sampling_args, native_tools, **extra_kwargs, **call_kwargs
        )

        choices = getattr(native, "choices", None) or []
        if len(choices) < m:
            return None  # server did not honor n=m -> fall back

        responses = []
        for k in range(m):
            single = native.model_copy(update={"choices": [choices[k]]})
            await client.raise_from_native_response(single)
            resp = await client.from_native_response(single)
            # Each block needs per-token data (Phase 4 pi_hat + interleave both read
            # response.message.tokens). If a choice lacks it, bail to the fallback.
            if getattr(resp.message, "tokens", None) is None:
                return None
            responses.append(resp)

        # Usage: native.usage covers the whole request (prompt + all m completions),
        # so increment ONCE -- the m-separate path's m single-completion increments
        # sum to the same total.
        try:
            self.increment_state_usage_from_response(state, responses[0])
        except Exception:  # noqa: BLE001 -- usage tracking is best-effort
            pass
        return responses

    # ------------------------------------------------------------------
    # Training-time probe: generation (Phase 1, D-4)
    # ------------------------------------------------------------------

    def _probe_block_sampling_args(self, state: vf.State, override: dict | None = None) -> dict:
        """Sampling args for the m probe blocks: the state's args (as the executed
        response uses them, T=0 under ProbeSamplingConfig) with temperature and
        extra_body.min_p replaced by the training values, so the pi_hat matrix is
        estimated under the distribution the trainer actually samples from."""
        base = dict(override if override is not None else (state.get("sampling_args") or {}))
        extra = dict(base.get("extra_body") or {})
        base["temperature"] = float(self.probe_block_temperature)
        extra["min_p"] = float(self.probe_block_min_p)
        base["extra_body"] = extra
        base.pop("n", None)
        return base

    async def _probe_get_model_response(
        self, state: vf.State, prompt: vf.Messages, parent_get_response, m: int, *args, **kwargs
    ):
        """Probe capture: TWO requests per turn, not one.

        1) The executed response with the state's sampling args (greedy under the
           probe config, mimicking eval). It is what the engine receives.
        2) m sampled reasoning blocks at the training temperature / min_p (one n=m
           request; m separate generations as fallback) for the pi_hat matrix.

        A single n=m+1 request cannot mix temperatures (plan Phase 1 §14). The stash
        contract matches the training path except ``_executed_block_idx`` is None:
        no sampled block executed, so _compute_pi_hat / _write_probe_extras use the
        plain mean over all m blocks instead of leave-one-out.
        """
        exec_resp = await parent_get_response(state, prompt, *args, **kwargs)

        block_args = self._probe_block_sampling_args(state, kwargs.get("sampling_args"))
        blocks = None
        if getattr(self, "use_nm_fusion", True):
            try:
                blocks = await self._generate_m_blocks_fused(state, prompt, m, sampling_args=block_args)
            except Exception as exc:  # noqa: BLE001 -- never fail the rollout on fusion
                logger.warning(
                    f"probe n=m block generation fusion failed ({exc!r}); falling back to "
                    "m separate generations (slow)."
                )
                blocks = None
        if blocks is None:
            block_kwargs = {**kwargs, "sampling_args": block_args}
            blocks = list(
                await asyncio.gather(
                    *[parent_get_response(state, prompt, *args, **block_kwargs) for _ in range(m)]
                )
            )

        state["_pending_reasoning_blocks"] = list(blocks)
        state["_executed_block_idx"] = None  # greedy response executed, not a sampled block
        return exec_resp

    async def setup_state(self, state: vf.State) -> vf.State:
        # Defensive: ensure TMPDIR exists before any planner subprocess.
        # fast_downward (invoked transitively by tw_env.reset()) does
        # tempfile.mkdtemp() + shutil.copy(libdownward.so, ...). If TMPDIR has
        # been deleted out from under us mid-run (observed live 2026-05-02 on
        # /workspace/tmp — root cause unknown but possibly wandb exit hooks or
        # a subprocess atexit handler), every planner setup fails permanently
        # and the orchestrator's retry loop spins forever. Idempotent mkdir is
        # microsecond-cheap and self-heals against this whole failure class.
        import tempfile as _tempfile
        os.makedirs(_tempfile.gettempdir(), exist_ok=True)

        game_file = state["info"]["game_file"]
        logger.debug(f"setup_state: loading {game_file}")
        tw_env = await asyncio.to_thread(self._make_tw_env, game_file)

        def _reset():
            with _logged_parse_lock("reset", game_file):
                return tw_env.reset()

        # Retry on transient OSError. The dominant case is MooseFS chunk-
        # visibility lag during fast_downward's per-call libdownward.so copy:
        # shutil.copyfile returns successfully, but when the planner subprocess
        # immediately dlopen's the .so, not all 34 MB of chunks are consistently
        # readable yet, so mmap of a PT_LOAD segment fails with "failed to map
        # segment from shared object". Observed live 2026-05-03 on the GRPO
        # run; manual cp+dlopen on the same path passes, confirming the failure
        # is timing-dependent. 3 attempts with linear backoff (0.5s, 1.0s, 1.5s)
        # covers the observed lag window. Non-OSError exceptions fall through
        # to the original error path (e.g., TextWorld parser failures).
        last_oserror = None
        success = False
        for attempt in range(3):
            try:
                obs, infos = await asyncio.to_thread(_reset)
                success = True
                break
            except OSError as exc:
                last_oserror = exc
                logger.warning(
                    f"setup_state: reset() OSError attempt {attempt + 1}/3 "
                    f"for {game_file!r}: {exc}. "
                    f"Retrying after {0.5 * (attempt + 1):.1f}s."
                )
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as exc:
                # Non-OSError - TextWorld 1.7.0 textgen parser failures, etc.
                # Re-raise as vf.Error so the framework catches it, records the
                # episode as failed (reward 0.0), and keeps the worker alive.
                logger.warning(f"setup_state: reset() failed for {game_file!r}: {exc}")
                try:
                    tw_env.close()
                except Exception:
                    pass
                raise vf.Error(f"setup_state failed for {game_file!r}: {exc}") from exc

        if not success:
            try:
                tw_env.close()
            except Exception:
                pass
            raise vf.Error(
                f"setup_state failed for {game_file!r} after 3 OSError retries: "
                f"{last_oserror}"
            ) from last_oserror

        state["alf_env"] = tw_env
        state["won"] = False
        # Truncation metrics — initialised here so they are always present in
        # state even for episodes where truncation never fires.
        state["context_truncated"] = False
        state["context_truncated_at_turn"] = None
        state["context_evictions"] = 0
        state["num_admissible"] = 0
        state["num_format_compliant"] = 0
        state["_last_admissible_commands"] = infos["admissible_commands"][0]
        if getattr(self, "oracle", "off") != "off":
            self._oracle_setup(state, game_file, obs[0], infos)
        initial_obs = self._format_obs(obs[0], infos["admissible_commands"][0])
        state["prompt"] = list(state["prompt"]) + [vf.UserMessage(content=initial_obs)]
        return state

    @vf.cleanup
    async def cleanup_alf_env(self, state: vf.State):
        tw_env = state.pop("alf_env", None)
        if tw_env is not None:
            try:
                tw_env.close()
            except Exception:
                pass
        # Action-level ARM (Phase 3): drop the transient m-block stash so it never
        # serializes into the RolloutOutput (it carries raw Response objects and is
        # consumed same-turn in Phase 4). Overwritten each turn; this is the
        # belt-and-suspenders end-of-rollout pop.
        state.pop("_pending_reasoning_blocks", None)
        state.pop("_executed_block_idx", None)
        # Training-time probe: the oracle agent is not serializable; the last
        # proposal was consumed by add_trajectory_step.
        state.pop("_probe_oracle_agent", None)
        state.pop("_probe_oracle_last", None)

    # ------------------------------------------------------------------
    # Training-time probe: oracle (Phase 1, D-6)
    # ------------------------------------------------------------------

    @staticmethod
    def _oracle_record(status: str, action: str | None = None, plan: list[str] | None = None) -> dict:
        return {
            "action": action,
            "status": status,
            "plan": plan,
            "steps_left": None if plan is None else len(plan),
        }

    def _oracle_setup(self, state: vf.State, game_file: str, intro_obs: str, infos) -> None:
        """Instantiate the in-env oracle right after reset(). Never raises.

        oracle="handcoded": alfworld's HandCodedTWAgent, driven directly rather than
        through the AlfredExpert TextWorld wrapper (which re-raises a timeout into
        the engine step and silently substitutes random actions; D-6). reset() reads
        traj_data.json next to the game file; observe(intro) records the room's
        receptacles. No proposal exists before the first action (status "init").
        oracle="planner": the engine's policy_commands from the requested EnvInfos.
        """
        state["_probe_oracle_agent"] = None
        state["_probe_oracle_last"] = dict(_ORACLE_OFF)
        oracle = getattr(self, "oracle", "off")
        try:
            if oracle == "handcoded":
                from alfworld.agents.expert import HandCodedTWAgent

                agent = HandCodedTWAgent(max_steps=200)
                agent.reset(game_file)
                agent.observe(intro_obs)
                state["_probe_oracle_agent"] = agent
                state["_probe_oracle_last"] = self._oracle_record("init")
            elif oracle == "planner":
                state["_probe_oracle_last"] = self._planner_oracle(infos)
        except Exception as exc:  # noqa: BLE001 -- the oracle must never fail a rollout
            logger.warning(f"oracle setup ({oracle}) failed for {game_file!r}: {exc!r}")
            state["_probe_oracle_last"] = self._oracle_record("error")

    def _planner_oracle(self, infos) -> dict:
        """Oracle record from the engine's policy_commands (oracle="planner")."""
        try:
            plan = [str(c) for c in (infos["policy_commands"][0] or [])]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"planner oracle: policy_commands unavailable: {exc!r}")
            return self._oracle_record("error")
        return self._oracle_record("planner", action=plan[0] if plan else None, plan=plan)

    def _oracle_after_step(self, state: vf.State, action: str, obs: str, infos) -> dict:
        """Oracle proposal for the state reached after executing ``action``: what the
        model should do on its next turn. Called from env_response after tw_env.step."""
        oracle = getattr(self, "oracle", "off")
        if oracle == "planner":
            return self._planner_oracle(infos)
        agent = state.get("_probe_oracle_agent")
        if agent is None:
            return self._oracle_record("error")
        if "facts" not in infos:
            # [VERIFY at Phase 5] the gym wrapper exposes EnvInfos.facts under this key.
            logger.warning("handcoded oracle: 'facts' missing from infos; reporting status=error")
            return self._oracle_record("error")
        game_state = {
            "feedback": obs,
            "admissible_commands": list(infos["admissible_commands"][0]),
            "facts": infos["facts"][0],
            "won": bool(infos["won"][0]) if "won" in infos else False,
        }
        return self._oracle_query(agent, game_state, action)

    def _oracle_query(self, agent, game_state: dict, last_action: str) -> dict:
        """One raw proposal from the handcoded policy plus a status.

        HandCodedAgent.act wraps policy.act with a random substitution (proposal ==
        last_action, or a "look" that led to an inadmissible proposal -> random
        admissible action). The probe wants the raw proposal AND whether that
        substitution would have fired, so policy.act is called exactly once (it
        advances the policy's internal step) and the two-line check is replicated
        here. Statuses (probe-schema.md §1): expert | random | look_fallback |
        failed | timeout | error.
        """
        try:
            raw = agent.policy.act(game_state, last_action)
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__  # alfworld's two NameError subclasses, by name
            if name == "HandCodedAgentTimeout":
                return self._oracle_record("timeout")
            if name == "HandCodedAgentFailed":
                return self._oracle_record("failed")
            logger.warning(f"handcoded oracle raised {exc!r}; reporting status=error")
            return self._oracle_record("error")
        admissible = game_state.get("admissible_commands") or []
        if raw == last_action or (last_action == "look" and raw not in admissible):
            status = "random"  # HandCodedAgent.act would substitute a random action
        elif raw not in admissible:
            status = "look_fallback"  # AlfredExpert would keep its default ["look"] plan
        else:
            status = "expert"
        return self._oracle_record(status, action=raw)

    def _parse_action(self, messages: vf.Messages) -> str:
        """Extract action text from the last AssistantMessage.

        Failure modes (deferred — revisit once baseline training runs are stable):

          - Unparsable response: falls back to "look" (always admissible, silent failure).
            Better behaviour: return None here and terminate the episode in env_response
            with reward 0.0, so format failures are visible to training.

          - Invalid action (not in admissible_commands): passed through to the game engine,
            which responds with "I don't understand that command." and wastes a turn.
            Natural signal for now; could add a small format penalty via the rubric later.
        """
        for msg in reversed(messages):
            if isinstance(msg, vf.AssistantMessage):
                content = msg.content or ""
                if not isinstance(content, str):
                    content = str(content)
                return self._parse_action_text(content)
        return "look"  # safe no-op if no model message found

    @staticmethod
    def _parse_action_text(content: str) -> str:
        """Action text from one assistant completion string: the per-message core of
        _parse_action, also applied to each sampled probe block (what that block
        *would* have executed). Same fallbacks, so a* matches the engine input."""
        # Primary: parse <action>...</action> tag
        match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: strip <think> block and use the rest
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return text if text else content.strip()

    @staticmethod
    def _completion_text(messages) -> str:
        """Content of the last assistant message in ``messages`` ("" if none)."""
        for msg in reversed(messages or []):
            if isinstance(msg, vf.AssistantMessage):
                content = msg.content or ""
                return content if isinstance(content, str) else str(content)
        return ""

    async def add_trajectory_step(self, state: vf.State, trajectory_step: vf.TrajectoryStep) -> None:
        """Action-level ARM (Phase 1): record this turn's admissible set + executed action.

        At this hook for turn k, ``state["_last_admissible_commands"]`` still holds the
        set the model saw when choosing this turn's action: env_response overwrites it
        only on the *next* turn's get_prompt_messages, which runs after this step is
        appended. The executed action is parsed from the just-generated completion via
        the same _parse_action used by env_response (so a* matches what the engine
        receives, including the "look" fallback on an unparsable completion).

        We write the *raw* admissible set + executed-action text into the step extras.
        prime-rl's interleave_rollout applies the union invariant
        (substitute_executed_into_admissible) and builds TrainingSample.decision_points
        from these keys. Keeping the union on the prime-rl side avoids duplicating the
        invariant across repos (verifiers does not depend on prime_rl) and mirrors how
        substitute_sampled_into_top_k is applied there for tokens.

        Cross-repo extras contract (consumed by prime_rl.orchestrator.trajectories):
            extras["admissible_actions"]: list[str]  # raw set, as the model saw it
            extras["executed_action"]:    str        # parsed executed action a*
            extras["pi_hat"]:             float       # action-level ARM only (Phase 4)
        """
        extras = trajectory_step.setdefault("extras", {})
        admissible = list(state.get("_last_admissible_commands", []))
        a_star = self._parse_action(trajectory_step["completion"])
        extras["admissible_actions"] = admissible
        extras["executed_action"] = a_star

        # Action-level ARM (Phase 4): teacher-force-score the admissible actions
        # against each stashed reasoning block (Phase 3), then leave-one-out
        # average the conditionals at a* to estimate the marginal pi_hat(a*|o).
        # Gated on m>1 (GRPO/PPO/token-ARM never enter this branch) and skipped on
        # error turns. Failures are non-fatal: pi_hat is left unset and the
        # decision point degrades to "no pi_hat" rather than killing the rollout.
        blocks = state.get("_pending_reasoning_blocks")
        j_star = state.get("_executed_block_idx")
        if getattr(self, "probe_capture", False):
            # Training-time probe (Phase 1): full matrix + window + oracle into
            # extras["probe"]; extras["pi_hat"] = plain-mean marginal at a*.
            await self._write_probe_extras(state, trajectory_step, extras, admissible, a_star, blocks)
        elif (
            self.num_reasoning_blocks > 1
            and blocks is not None
            and j_star is not None
            and state.get("error") is None
        ):
            try:
                extras["pi_hat"] = await self._compute_pi_hat(
                    state, blocks, j_star, admissible, a_star
                )
            except Exception as exc:
                logger.warning(f"pi_hat computation failed; leaving it unset: {exc}")

        # Drop the transient stash (consumed here; cleanup_alf_env pops it again
        # defensively at rollout end).
        state.pop("_pending_reasoning_blocks", None)
        state.pop("_executed_block_idx", None)

        await super().add_trajectory_step(state, trajectory_step)

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _action_continuation_ids(self, reasoning_text: str, action_text: str) -> list[int]:
        """Token-ids of "<action>{action}</action>" as a continuation of the block's
        reasoning, via in-context tokenization (encode the full text, slice off the
        reasoning's common-prefix tokens). Slicing handles a BPE merge at the
        </think>-><action> seam. The closing </action> terminator is load-bearing:
        without it "go to cabinet 1" is a strict token-prefix of "go to cabinet 12"
        and the shorter action gets a systematically inflated score (phase doc §14).
        """
        full = self._tokenizer.encode(
            reasoning_text + "<action>" + action_text + "</action>",
            add_special_tokens=False,
        )
        base = self._tokenizer.encode(reasoning_text, add_special_tokens=False)
        cpl = self._common_prefix_len(base, full)
        return full[cpl:]

    @staticmethod
    def _admissible_union(admissible: list[str], a_star: str) -> tuple[list[str], int]:
        """Union invariant: a* is among the scored candidates (appended at the end if
        absent). Mirrors prime_rl.orchestrator.admissible.substitute_executed_into_admissible
        (replicated here because verifiers does not depend on prime_rl; a test pins the
        two orders equal) -- the index space of every per-action vector downstream."""
        candidates = list(admissible)
        if a_star in candidates:
            return candidates, candidates.index(a_star)
        candidates.append(a_star)
        return candidates, len(candidates) - 1

    async def _score_pi_hat_matrix(
        self,
        state: vf.State,
        blocks: list,
        admissible: list[str],
        a_star: str,
    ):
        """Teacher-force-score every admissible action against every reasoning block.

        For each block R_j: build [o, R_j, <action>a</action>] for every candidate a
        (re-encoding reasoning+action together so the </think>-><action> BPE seam is
        handled, and scoring from the first divergent token so every candidate shares
        the same conditioning prefix -- the shared seam token cancels in the softmax).
        Score the whole m x |A| batch in one request, renormalize each block's row
        over the candidates to get pi(·|o, R_j). Scored at T=1 / neutral so the
        conditionals are the raw policy.

        Returns a PiHatMatrix (rbmc.py): candidates (admissible union), a_star_idx,
        cond[m][|A|] (softmax over the union when self.pi_hat_renorm, else the raw
        sequence probabilities exp(logp); D-13), Z[m] (per-block log-sum-exp, the
        admissible mass whether or not it was applied), raw[m][|A|]. The
        training path reduces it to the leave-one-out scalar at a* (_compute_pi_hat);
        the probe capture path records all of it (_write_probe_extras).
        """
        # rbmc ships alongside this module (pyproject include). Dual import covers
        # both the installed top-level layout and the in-repo package layout.
        try:
            from rbmc import PiHatMatrix, conditional_renorm, logsumexp, raw_probs
        except ImportError:
            from environments.alfworld.rbmc import (
                PiHatMatrix,
                conditional_renorm,
                logsumexp,
                raw_probs,
            )

        self._ensure_tokenizer(state)

        candidates, a_star_idx = self._admissible_union(admissible, a_star)

        client = state["client"]
        model = state["model"]

        # Phase 9 perf (CRITICAL): build ALL m x |A| scoring prompts and score them
        # in ONE /v1/completions request, instead of one request per block. The
        # per-block requests were an m x request multiplier that flooded the vLLM
        # scheduler (300-deep prefill queue, GPU ~idle).
        #
        # Each prompt is o_ids + encode(reasoning_j + <action>a</action>), so within
        # a block the |A| prompts share the contiguous [o, R_j] prefix (~2000 tok)
        # and across blocks they share [o]. BUT a single batched request prefills
        # all its prompts *simultaneously* within a scheduler step, so [o, R_j] is
        # NOT cached before its |A| action prompts hit it -- each action would
        # re-prefill the full ~2000-token prefix (~|A|x redundancy; observed as a
        # 56% -> 31% prefix-cache hit-rate drop and a sustained-prefill wall).
        #
        # Fix: PRE-WARM each [o, R_j] prefix with a cheap forward FIRST (one batched
        # request, awaited to completion), so the prefix is resident in the cache
        # when the batched action scoring arrives. Each action prompt then reuses
        # [o, R_j] and pays only its ~20-token action suffix. [o] itself is already
        # hot from this turn's generation. (REQUIRES prefix caching on + a generous
        # max_num_batched_tokens so the batched prefill is admitted, not serialized.)
        prewarm_prefixes: list[list[int]] = []
        all_prompts: list[list[int]] = []
        all_spans: list[tuple[int, int]] = []
        block_slices: list[tuple[int, int]] = []  # [start, end) into all_prompts per block
        for block in blocks:
            o_ids = list(block.message.tokens.prompt_ids)
            content = block.message.content or ""
            reasoning_text = content.split("<action>")[0]
            base_ids = self._tokenizer.encode(reasoning_text, add_special_tokens=False)
            prewarm_prefixes.append(o_ids + base_ids)  # [o, R_j] -- the shared prefix

            block_start = len(all_prompts)
            for a in candidates:
                full = self._tokenizer.encode(
                    reasoning_text + "<action>" + a + "</action>",
                    add_special_tokens=False,
                )
                cpl = self._common_prefix_len(base_ids, full)
                all_prompts.append(o_ids + full)
                all_spans.append((len(o_ids) + cpl, len(o_ids) + len(full)))
            block_slices.append((block_start, len(all_prompts)))

        # Pre-warm the [o, R_j] prefixes into the prefix cache before scoring, so the
        # batched action prompts reuse them instead of recomputing per action.
        # Best-effort: a failure just means scoring pays the full prefill (slower).
        if getattr(self, "prewarm_pi_hat_prefixes", True):
            try:
                await self._prewarm_prefixes(client, prewarm_prefixes, model)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"pi_hat prefix pre-warm skipped: {exc!r}")

        # Single request for the whole decision point's m x |A| prompts (now mostly
        # prefix-cache hits on [o, R_j]).
        logps = await self._score_continuations(client, all_prompts, all_spans, model)

        renorm = bool(getattr(self, "pi_hat_renorm", True))
        cond: list[list[float]] = []
        Z: list[float] = []
        raw: list[list[float]] = []
        for start, end in block_slices:
            row = list(logps[start:end])
            raw.append(row)
            cond.append(conditional_renorm(row) if renorm else raw_probs(row))
            Z.append(logsumexp(row))
        return PiHatMatrix(
            candidates=candidates, a_star_idx=a_star_idx, cond=cond, Z=Z, raw=raw, renormalized=renorm
        )

    async def _score_pi_hat_astar(
        self,
        state: vf.State,
        blocks: list,
        a_star: str,
        skip_idx: int | None = None,
    ) -> list[float]:
        """Teacher-force-score ONLY the executed action a* against the reasoning
        blocks the estimator uses (every block except ``skip_idx``).

        The training-path shortcut for pi_hat_renorm=False (D-13, 2026-11-07).
        Without the softmax over the admissible union, block j's value at a* is
        the raw sequence probability exp(logp(<action>a*</action> | o, R_j)) and
        depends on no other candidate, so the other |A|-1 prompts per block are
        dead weight -- and so is block j* itself, which the leave-one-out mean
        drops. Each prompt is built exactly as in _score_pi_hat_matrix (o_ids +
        the re-encoded reasoning_j + "<action>a*</action>", scored from the first
        token that diverges from the bare reasoning, so the </think>-><action>
        BPE seam is handled identically) and the whole decision point goes out
        in ONE _score_continuations call, after the same best-effort [o, R_j]
        pre-warm (prewarm_pi_hat_prefixes). Note the pre-warm buys nothing here:
        with a single prompt per prefix there is no in-batch redundancy to
        remove, so it only adds the prefill it was meant to share.

        Returns rbmc.raw_probs of the per-block span logprobs, in block order
        with the skipped block absent (len(blocks) - 1 values when skip_idx is a
        valid index, len(blocks) otherwise).
        """
        try:
            from rbmc import raw_probs
        except ImportError:
            from environments.alfworld.rbmc import raw_probs

        self._ensure_tokenizer(state)
        client = state["client"]
        model = state["model"]

        prewarm_prefixes: list[list[int]] = []
        prompts: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for j, block in enumerate(blocks):
            if j == skip_idx:
                continue
            o_ids = list(block.message.tokens.prompt_ids)
            content = block.message.content or ""
            reasoning_text = content.split("<action>")[0]
            base_ids = self._tokenizer.encode(reasoning_text, add_special_tokens=False)
            prewarm_prefixes.append(o_ids + base_ids)  # [o, R_j] -- the shared prefix
            full = self._tokenizer.encode(
                reasoning_text + "<action>" + a_star + "</action>",
                add_special_tokens=False,
            )
            cpl = self._common_prefix_len(base_ids, full)
            prompts.append(o_ids + full)
            spans.append((len(o_ids) + cpl, len(o_ids) + len(full)))

        if getattr(self, "prewarm_pi_hat_prefixes", True):
            try:
                await self._prewarm_prefixes(client, prewarm_prefixes, model)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"pi_hat prefix pre-warm skipped: {exc!r}")

        logps = await self._score_continuations(client, prompts, spans, model)
        return raw_probs(list(logps))

    async def _compute_pi_hat(
        self,
        state: vf.State,
        blocks: list,
        j_star: int | None,
        admissible: list[str],
        a_star: str,
    ) -> float:
        """RBMC estimate of pi_hat(a*|o) from the m reasoning blocks.

        Training path: j_star is the block that generated a*; average the
        conditionals at a* across blocks leaving it out (rbmc_marginal_loo, the
        hold-out structure of RLOO). Probe capture: j_star is None (the executed
        response was a separate greedy generation, no sampled block produced a*), so
        the plain mean over all m blocks is the right estimate (rbmc_marginal).

        Scoring cost per decision point. With pi_hat_renorm=True the softmax over
        the admissible union needs every candidate, so the full m x |A| matrix is
        scored (_score_pi_hat_matrix). With pi_hat_renorm=False (D-13) the value at
        a* is the raw sequence probability and depends on no other candidate, so
        only "<action>a*</action>" is scored, and only under the blocks the
        estimator uses: m - 1 prompts (j* skipped) on the training path, m when
        j_star is None (_score_pi_hat_astar). The reduction is the same in both
        cases; the unscored j* slot holds 0.0, which rbmc_marginal_loo subtracts
        back out exactly, so the estimate equals the matrix path's to rounding.
        probe_capture=True always keeps the full matrix (the record needs it).
        """
        try:
            from rbmc import rbmc_marginal, rbmc_marginal_loo
        except ImportError:
            from environments.alfworld.rbmc import rbmc_marginal, rbmc_marginal_loo

        renorm = bool(getattr(self, "pi_hat_renorm", True))
        if not renorm and not getattr(self, "probe_capture", False):
            scored = iter(await self._score_pi_hat_astar(state, blocks, a_star, skip_idx=j_star))
            conditionals_at_astar = [0.0 if j == j_star else next(scored) for j in range(len(blocks))]
        else:
            mat = await self._score_pi_hat_matrix(state, blocks, admissible, a_star)
            conditionals_at_astar = [row[mat.a_star_idx] for row in mat.cond]
        if j_star is None:
            return rbmc_marginal(conditionals_at_astar)
        return rbmc_marginal_loo(conditionals_at_astar, exclude_idx=j_star)

    async def _write_probe_extras(
        self,
        state: vf.State,
        trajectory_step,
        extras: dict,
        admissible: list[str],
        a_star: str,
        blocks: list | None,
    ) -> None:
        """Training-time probe capture: write extras["probe"] (probe-schema.md §1) and
        extras["pi_hat"] (plain-mean marginal at a*). Never raises. On a scoring
        failure the record is still written with blocks=[] / pi_hat_marginal=None /
        Z=[] and extras["pi_hat"] left unset, mirroring the training path's degrade
        (the server then zeroes adv_star for the turn)."""
        completion_text = self._completion_text(trajectory_step.get("completion") or [])
        step_idx = len(state.get("trajectory") or [])  # this turn's index; step appended after
        candidates, a_star_idx = self._admissible_union(admissible, a_star)

        prompt_msgs = trajectory_step.get("prompt") or []
        window: dict[str, Any] = {
            "labels": [],
            "token_count": None,
            "evictions_so_far": int(state.get("context_evictions", 0) or 0),
        }
        try:
            window["labels"] = self._label_prompt(prompt_msgs, step_idx)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"probe window labels failed: {exc!r}")
        try:
            window["token_count"] = int(await self._count_tokens(prompt_msgs, state))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"probe window token count failed: {exc!r}")

        oracle = state.get("_probe_oracle_last") or _ORACLE_OFF
        probe: dict[str, Any] = {
            "schema": PROBE_SCHEMA_VERSION,
            "pi_hat_renorm": bool(getattr(self, "pi_hat_renorm", True)),
            "admissible_union": candidates,
            # the raw admissible count: candidates[n_admissible:] is the executed
            # action appended by the union invariant (an inadmissible a*)
            "n_admissible": len(admissible),
            "executed_idx": a_star_idx,
            "parsed_ok": re.search(r"<action>.*?</action>", completion_text, re.DOTALL) is not None,
            "exec_block": {
                "reasoning": completion_text.split("<action>")[0],
                "action": a_star,
                "completion": completion_text,
            },
            "blocks": [],
            "pi_hat_marginal": None,
            "Z": [],
            "raw_logps": None,
            "window": window,
            "oracle": {k: oracle.get(k) for k in _ORACLE_FIELDS},
        }

        if blocks and state.get("error") is None:
            try:
                mat = await self._score_pi_hat_matrix(state, blocks, admissible, a_star)
                if mat.candidates != candidates:  # same construction; guard the invariant
                    raise RuntimeError("admissible union mismatch between record and matrix")
                marginal = mat.marginal()
                rows = []
                for j, block in enumerate(blocks):
                    content = block.message.content or ""
                    rows.append(
                        {
                            "reasoning": content.split("<action>")[0],
                            "action": self._parse_action_text(content),
                            "cond": mat.cond[j],
                        }
                    )
                probe["blocks"] = rows
                probe["pi_hat_marginal"] = marginal
                probe["Z"] = mat.Z
                if getattr(self, "store_raw_logps", False):
                    probe["raw_logps"] = mat.raw
                extras["pi_hat"] = marginal[mat.a_star_idx]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"probe pi_hat matrix failed; writing degraded record: {exc!r}")

        extras["probe"] = probe

    async def _prewarm_prefixes(self, client, prefix_token_ids: list[list[int]], model: str) -> None:
        """Prefill the [o, R_j] prefixes (one per reasoning block) so the subsequent
        batched action scoring reuses them from vLLM's prefix cache.

        A cheap 1-token completion still prefills (and thus caches) the whole prompt;
        we discard the output. Awaited to completion so the cache is populated before
        the scoring request is issued. No prompt_logprobs here -- this is pure cache
        population. [o] is already hot from this turn's generation, so each prefix
        only pays its own reasoning block's prefill (inherent), once instead of |A|x.
        """
        if not prefix_token_ids:
            return
        body = dict(
            model=model,
            prompt=prefix_token_ids,  # vLLM accepts list[list[int]]
            max_tokens=1,
            temperature=0.0,
        )
        await client.token_client.post(
            "/v1/completions", body=body, cast_to=CompletionWithPromptLogprobs
        )

    async def _score_continuations(
        self,
        client,
        prompts_token_ids: list[list[int]],
        spans: list[tuple[int, int]],
        model: str,
        temperature: float = 1.0,
    ) -> list[float]:
        """Return the summed actual-token logprob over each prompt's span.

        Phase 9 fix: prefer the custom /v1/score route, which computes the span sums
        SERVER-side and returns one float per prompt -- avoiding the full-prompt
        prompt_logprobs payload (len(prompt) logprob dicts per prompt) that was the
        cache-invariant throughput wall. Falls back to the /v1/completions
        prompt_logprobs path if /v1/score is unavailable (older inference server),
        so correctness is preserved either way.
        """
        if getattr(self, "use_score_route", True):
            try:
                return await self._score_via_route(
                    client, prompts_token_ids, spans, model, temperature
                )
            except Exception as exc:  # noqa: BLE001 -- fall back, never fail scoring
                logger.warning(
                    f"/v1/score failed ({exc!r}); falling back to /v1/completions "
                    "prompt_logprobs (full-payload, slower)."
                )
        return await self._score_via_completions(
            client, prompts_token_ids, spans, model, temperature
        )

    async def _score_via_route(
        self,
        client,
        prompts_token_ids: list[list[int]],
        spans: list[tuple[int, int]],
        model: str,
        temperature: float,
    ) -> list[float]:
        """POST to the custom /v1/score route (server-side span summation -> tiny
        response: one float per prompt)."""
        body = dict(
            model=model,
            prompts=prompts_token_ids,
            spans=[[int(s), int(e)] for s, e in spans],
            temperature=temperature,
            top_p=1.0,
        )
        resp = await client.token_client.post("/v1/score", body=body, cast_to=ScoreResponse)
        scores = list(resp.scores)
        if len(scores) != len(prompts_token_ids):
            raise ValueError(
                f"/v1/score returned {len(scores)} scores for {len(prompts_token_ids)} prompts"
            )
        return scores

    async def _score_via_completions(
        self,
        client,
        prompts_token_ids: list[list[int]],
        spans: list[tuple[int, int]],
        model: str,
        temperature: float = 1.0,
    ) -> list[float]:
        """Fallback scorer via native /v1/completions prompt_logprobs (full per-position
        payload). The actual token at each scored position is known by construction,
        so logprobs are read by token-id. One value per prompt, aligned to spans."""
        body = _build_score_body(prompts_token_ids, model, temperature)
        resp = await client.token_client.post(
            "/v1/completions", body=body, cast_to=CompletionWithPromptLogprobs
        )
        out: list[float] = [0.0] * len(prompts_token_ids)
        for choice in resp.choices:
            start, end = spans[choice.index]
            out[choice.index] = _span_logprob_sum(
                choice.prompt_logprobs, prompts_token_ids[choice.index], start, end
            )
        return out

    @staticmethod
    def _label_prompt(messages, step_idx: int) -> list[str]:
        """Assign global turn labels to the post-eviction messages for trajectory step step_idx.

        Labels are derived by counting backwards from the end of the message list.
        At step k (0-indexed), the last user message is User (Turn k) and the last
        assistant message is Assistant (Turn k).
        """
        user_count = step_idx + 1  # last user msg = User (Turn step_idx+1)
        asst_count = step_idx      # last asst msg = Assistant (Turn step_idx)
        labels = []
        for msg in reversed(messages):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "user")
            if role == "system":
                labels.append("System Prompt")
            elif role == "user":
                labels.append(f"User (Turn {user_count})")
                user_count -= 1
            elif role == "assistant":
                labels.append(f"Assistant (Turn {asst_count})")
                asst_count -= 1
        labels.reverse()
        return labels

    @staticmethod
    def _write_trajectory_log(state: vf.State) -> None:
        """Write a human-readable trajectory log annotating each assistant turn with its context window."""
        game_file = state.get("info", {}).get("game_file", "unknown")
        task_name = Path(game_file).parent.name
        outcome = "win" if state.get("won") else "loss"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_dir = Path.cwd() / "trajectories"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{task_name}_{outcome}_{timestamp}.txt"

        def _content(msg) -> str:
            c = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            return str(c or "")

        lines: list[str] = []

        # Initial prompt: system message + initial observation
        for msg in state.get("prompt", []):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "user")
            if role == "system":
                lines += ["==== System Prompt", _content(msg), ""]
            elif role == "user":
                lines += ["==== User (Turn 1)", _content(msg), ""]

        # One TrajectoryStep per LLM call
        trajectory = state.get("trajectory", [])
        for k, step in enumerate(trajectory):
            ctx_labels = ALFWorldEnvironment._label_prompt(step["prompt"], k)
            asst_content = _content(step["completion"][0]) if step["completion"] else ""
            lines += [
                f"==== Assistant (Turn {k + 1})",
                f"Context = [{', '.join(ctx_labels)}]",
                asst_content,
                "",
            ]
            # Env response: last message of the next step's prompt, or the terminal response,
            # or absent if the episode ended by hitting max_turns without a game response.
            if k + 1 < len(trajectory):
                env_content = _content(trajectory[k + 1]["prompt"][-1])
                lines += [f"==== User (Turn {k + 2})", env_content, ""]
            else:
                final_env_response = state.get("final_env_response")
                if final_env_response:
                    lines += [f"==== User (Turn {k + 2})", _content(final_env_response[0]), ""]

        log_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Trajectory log written: {log_path}")

    async def render_completion(self, state: vf.State) -> None:
        await super().render_completion(state)
        if self.log_trajectories == "none":
            return
        won = state.get("won", False)
        should_log = self.log_trajectories == "all" or (self.log_trajectories == "wins" and won)
        if should_log:
            self._write_trajectory_log(state)

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        tw_env = state["alf_env"]
        action = self._parse_action(messages)

        # Compare against the commands the model saw at choice time, not the post-step set.
        if action in state.get("_last_admissible_commands", []):
            state["num_admissible"] += 1

        # Track format compliance: did the latest assistant message contain BOTH
        # <think> and <action> tags? Strict format check (parallel to
        # admissible_action_rate). Used as a zero-weight metric to detect
        # format-decay during long-context generation or RL-induced schema drift.
        for msg in reversed(messages):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if isinstance(content, str) and "<think>" in content and "<action>" in content:
                    state["num_format_compliant"] += 1
                break

        def _step():
            with _TW_PARSE_LOCK:
                return tw_env.step([action])

        # Guard against fast_downward planner hangs: a specific game/state can wedge the
        # planner subprocess inside tw_env.step and (via the shared _TW_PARSE_LOCK) stall
        # every rollout in the worker indefinitely -- invisible to the router heartbeat,
        # which runs on the free event loop. Time the step out so the rollout errors and is
        # dropped/retried instead of hanging evaluate/collect forever. Normal steps are <1s.
        #
        # Steps are NOT wrapped in _logged_parse_lock (would emit ~15 lines/rollout of
        # noise that buries the reset-wedge signal). Instead we surface only the anomalies:
        # a >120s timeout (this step, or a reset wedge holding the lock ahead of it) and a
        # merely-slow step. Fast steps stay silent.
        game_file = state.get("info", {}).get("game_file", "?")
        _t_step = time.monotonic()
        try:
            obs, _reward, done, infos = await asyncio.wait_for(
                asyncio.to_thread(_step), timeout=120.0
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[lock] STEP-TIMEOUT (>120s) game={game_file} pid={os.getpid()} "
                f"action={action!r} -- step blocked, likely behind a wedged reset "
                f"holding _TW_PARSE_LOCK (cross-check the [lock] HOLDING lines)."
            )
            raise
        _dt_step = time.monotonic() - _t_step
        if _dt_step > 5.0:
            logger.warning(
                f"[lock] SLOW-STEP game={game_file} pid={os.getpid()} "
                f"took={_dt_step:.2f}s action={action!r}"
            )
        state["_last_admissible_commands"] = infos["admissible_commands"][0]
        if getattr(self, "oracle", "off") != "off":
            # Training-time probe: the oracle's proposal for the state the model
            # acts on next turn (consumed by add_trajectory_step, probe capture).
            state["_probe_oracle_last"] = self._oracle_after_step(state, action, obs[0], infos)

        content = self._format_obs(obs[0], infos["admissible_commands"][0])
        response = vf.UserMessage(content=content)

        if done[0]:
            state["won"] = bool(infos["won"][0])
            state["final_env_response"] = [response]

        return [response]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def alfworld_reward(state: vf.State, **kwargs) -> float:
    return 1.0 if state.get("won", False) else 0.0


def context_truncated(state: vf.State, **kwargs) -> float:
    """Zero-weight metric: exposes state["context_truncated"] to W&B via rubric."""
    return float(state.get("context_truncated", False))


def context_evictions(state: vf.State, **kwargs) -> float:
    """Zero-weight metric: exposes state["context_evictions"] to W&B via rubric."""
    return float(state.get("context_evictions", 0))


def admissible_action_rate(state: vf.State, **kwargs) -> float:
    """Zero-weight metric: fraction of decisions whose parsed action was in the admissible set."""
    n = len(state.get("trajectory", []))
    return state.get("num_admissible", 0) / n if n > 0 else 0.0


def format_compliance_rate(state: vf.State, **kwargs) -> float:
    """Zero-weight metric: fraction of decisions whose assistant message contained
    BOTH <think> and <action> tags.

    Strict format check — parallel to admissible_action_rate but at the schema
    level rather than the action-validity level. A drop in this metric during
    GRPO indicates the model is regressing on the XML schema the SFT phase
    established (Phase 4 SFT baseline: 100% compliance at every measured turn).
    """
    n = len(state.get("trajectory", []))
    return state.get("num_format_compliant", 0) / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

# ALFWorld task types — used to validate curriculum config and as the canonical set.
KNOWN_TASK_TYPES: frozenset = frozenset([
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
])


def _task_type_from_game_file(game_file: str) -> str:
    """Extract ALFWorld task type from a game-file path.

    Path layout: .../<split>/<task_type>-<obj>-<recep>-<id>/<trial_dir>/game.tw-pddl
    The task type is the prefix (before the first "-") of the second-to-last
    directory component.

    Returns the task type string (e.g. "look_at_obj_in_light").
    """
    parent = os.path.basename(os.path.dirname(os.path.dirname(game_file)))
    return parent.split("-", 1)[0]


def _curriculum_weights_for_task_type(
    task_type: str,
    curriculum: list,
) -> list:
    """Compute per-stage sampling weights for a task type.

    For each stage in the curriculum, returns the weight assigned to this task
    type. Falls back to stage["default"] if the task type is not explicitly
    listed, else 0.0.

    Args:
        task_type: Task type string (e.g. "look_at_obj_in_light").
        curriculum: List of stage dicts. Each stage maps task type names to
            relative weights (float >= 0). Optional "default" key applies to
            any task type not explicitly listed.

    Returns:
        List of floats — one per stage.
    """
    weights = []
    for stage_idx, stage in enumerate(curriculum):
        if not isinstance(stage, dict):
            raise ValueError(
                f"curriculum[{stage_idx}] must be a dict mapping task type to weight, "
                f"got {type(stage).__name__}"
            )
        if task_type in stage:
            weights.append(float(stage[task_type]))
        elif "default" in stage:
            weights.append(float(stage["default"]))
        else:
            weights.append(0.0)
    return weights


def _build_probe_dataset(
    data_path: str,
    probe_games: list[dict],
    curriculum: list | None = None,
    task_types: list[str] | None = None,
) -> Dataset:
    """Rows for exactly the probe set's games, in list order (training-probe D-5).

    Each ``info`` carries ``game_file`` (resolved under data_path; existence is
    checked at setup_state, as for scanned rows) plus the probe-set identity the
    orchestrator writes into every record: ``game_id``, ``probe_id``, ``split``,
    ``task_type``, ``role``. The probe set fixes the list, so curriculum /
    task_types are ignored (with a warning) when given.
    """
    try:
        from probe_sets import resolve_game_file
    except ImportError:
        from environments.alfworld.probe_sets import resolve_game_file

    if not probe_games:
        raise ValueError("probe_games is empty")
    if curriculum is not None or task_types is not None:
        logger.warning("probe_games given: curriculum / task_types are ignored (the probe set fixes the game list)")
    rows = []
    for g in probe_games:
        rows.append(
            {
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
                "answer": "",
                "info": {
                    "game_file": resolve_game_file(data_path, g["game_id"]),
                    "game_id": g["game_id"],
                    "probe_id": g["probe_id"],
                    "split": g["split"],
                    "task_type": g["task_type"],
                    "role": g["role"],
                },
            }
        )
    return Dataset.from_list(rows)


def build_dataset(
    data_path: str,
    split: str,
    curriculum: list | None = None,
    task_types: list[str] | None = None,
    probe_games: list[dict] | None = None,
) -> Dataset:
    """Scan game files under data_path/split and return a HuggingFace Dataset.

    If `curriculum` is provided, each row is annotated with a top-level
    `curriculum_weights` field — a list of per-stage weights derived from the
    row's task type. The Buffer reads `row["curriculum_weights"][stage]` to
    compute the sampling probability for the row at the current stage.

    Args:
        data_path: Path to ALFWorld data root (containing split subdirectories).
        split: Dataset split ("train" / "valid_seen" / "valid_unseen").
        curriculum: Optional curriculum spec. List of stage dicts mapping task
            type names (and optionally "default") to relative weights.

    Returns:
        HuggingFace Dataset with rows containing prompt, answer, info, and
        (if curriculum is set) curriculum_weights.
        probe_games: Training-time probe. When given (the loaded probe-set game
            list), build rows for exactly those games in order instead of
            scanning ``split``; see _build_probe_dataset.
    """
    if probe_games is not None:
        return _build_probe_dataset(data_path, probe_games, curriculum=curriculum, task_types=task_types)

    split_map = {
        "train": "train",
        "valid_seen": "valid_seen",
        "valid_unseen": "valid_unseen",
    }
    base_path = os.path.join(data_path, split_map.get(split, split))

    game_files = []
    for root, _dirs, files in os.walk(base_path):
        for fname in files:
            if fname == "game.tw-pddl":
                game_files.append(os.path.join(root, fname))
    game_files.sort()

    # Optional task-type filter (e.g. task_types=["pick_and_place_simple"]) -> restrict
    # collect / RL to a single ALFWorld type. Uses the same path->type parse as the
    # curriculum weighting, so it composes with (or replaces) the curriculum.
    if task_types is not None:
        allowed = set(task_types)
        game_files = [gf for gf in game_files if _task_type_from_game_file(gf) in allowed]

    if not game_files:
        raise ValueError(
            f"No game.tw-pddl files found under {base_path}"
            + (f" for task_types={task_types}" if task_types is not None else "")
        )

    if curriculum is not None:
        # Validate referenced task types match known set; warn on unknowns.
        referenced = set()
        for stage in curriculum:
            if isinstance(stage, dict):
                referenced.update(k for k in stage.keys() if k != "default")
        unknown = referenced - KNOWN_TASK_TYPES
        if unknown:
            logger.warning(
                f"Curriculum references task types not in the known ALFWorld set: "
                f"{sorted(unknown)}. These weights will not match any rows. "
                f"Known types: {sorted(KNOWN_TASK_TYPES)}"
            )

    rows = []
    for gf in game_files:
        row = {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
            "answer": "",
            "info": {"game_file": gf},
        }
        if curriculum is not None:
            task_type = _task_type_from_game_file(gf)
            row["curriculum_weights"] = _curriculum_weights_for_task_type(task_type, curriculum)
        rows.append(row)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Entry point called by vf.load_environment("alfworld-env", **args)
# ---------------------------------------------------------------------------

def load_environment(
    data_path: str = os.path.expanduser("~/.cache/alfworld/json_2.1.1"),
    split: str = "train",
    max_turns: int = MAX_EPISODE_STEPS,
    max_context_tokens: int = -1,
    log_trajectories: str = "none",
    curriculum: list | None = None,
    task_types: list[str] | None = None,
    tokenizer_path: str | None = None,
    num_reasoning_blocks: int = 1,
    use_nm_fusion: bool = True,
    prewarm_pi_hat_prefixes: bool = True,
    use_score_route: bool = True,
    probe_set: str | None = None,
    probe_capture: bool = False,
    probe_block_temperature: float = 0.8,
    probe_block_min_p: float = 0.1,
    oracle: str = "off",
    store_raw_logps: bool = False,
    pi_hat_renorm: bool = True,
) -> vf.Environment:
    """
    Args:
        tokenizer_path: Path/HF-id for the tokenizer used by the context-window
            truncation logic. When set, overrides state["model"]. **REQUIRED**
            for any LoRA-enabled run that resumes from a checkpoint: vLLM
            advertises the LoRA adapter NAME (not a real path) in those setups,
            and AutoTokenizer.from_pretrained on that name fails. Set this to
            the base model path (e.g. "/workspace/sft-checkpoint" or
            "Qwen/Qwen2.5-1.5B-Instruct"). Tokenization is LoRA-invariant.
        probe_set: Training-time probe (plan/training-probe). Name of a probe set
            shipped in probe_sets/ (e.g. "alfworld-24-v1"); the dataset is then
            exactly those games, in order, and ``split`` is not scanned.
        probe_capture: Greedy executed response + m sampled blocks per turn and
            extras["probe"] per step (probe-schema.md §1). Requires
            num_reasoning_blocks >= 2. Off on the training path.
        probe_block_temperature / probe_block_min_p: sampling for the m probe
            blocks (the training values, 0.8 / 0.1 by default).
        oracle: "off" | "handcoded" (alfworld HandCodedTWAgent in-env) | "planner"
            (engine policy_commands; slow, fast_downward per step).
        store_raw_logps: keep the raw m x |A| span logprobs in the probe record.
        pi_hat_renorm: True = per-block conditionals are the softmax over the
            admissible union (the pre-2026-11-07 convention); False = raw sequence
            probabilities, no renormalization (training-probe D-13). Applies to
            the training path and to probe capture alike.
    """
    rubric = vf.Rubric()
    rubric.add_reward_func(alfworld_reward, weight=1.0)
    rubric.add_reward_func(context_truncated, weight=0.0)
    rubric.add_reward_func(context_evictions, weight=0.0)
    rubric.add_reward_func(admissible_action_rate, weight=0.0)
    rubric.add_reward_func(format_compliance_rate, weight=0.0)

    probe_games = None
    if probe_set is not None:
        try:
            from probe_sets import load_probe_set
        except ImportError:
            from environments.alfworld.probe_sets import load_probe_set
        probe_games = load_probe_set(probe_set)

    env = ALFWorldEnvironment(
        dataset=lambda: build_dataset(
            data_path, split, curriculum=curriculum, task_types=task_types, probe_games=probe_games
        ),
        rubric=rubric,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        log_trajectories=log_trajectories,
        tokenizer_path=tokenizer_path,
        num_reasoning_blocks=num_reasoning_blocks,
        use_nm_fusion=use_nm_fusion,
        prewarm_pi_hat_prefixes=prewarm_pi_hat_prefixes,
        use_score_route=use_score_route,
        probe_capture=probe_capture,
        probe_block_temperature=probe_block_temperature,
        probe_block_min_p=probe_block_min_p,
        oracle=oracle,
        store_raw_logps=store_raw_logps,
        pi_hat_renorm=pi_hat_renorm,
        env_id="alfworld-env",
    )
    return env
