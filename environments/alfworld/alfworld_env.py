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
"""
import asyncio
import datetime
import logging
import os
import random
import re
import threading
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

# Per-process cache: game_file -> tw_env_id registered with textworld.gym.
# Populated lazily on first rollout of each game file; lock only needed on
# cache miss (i.e. first time a given game file is seen in this process).
_TW_ENV_ID_CACHE: dict[str, str] = {}
_TW_REGISTER_LOCK = threading.Lock()
_TW_PARSE_LOCK = threading.Lock()


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
        extra_body=dict(prompt_logprobs=1),
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

    # ------------------------------------------------------------------
    # TextWorld helpers
    # ------------------------------------------------------------------

    def _get_tw_env_id(self, game_file: str) -> str:
        """Return the textworld gym id for game_file, registering it if needed.

        Registration is done at most once per game file per process (lazy cache).
        The lock is only contended on the first rollout of each game file.
        """
        if game_file in _TW_ENV_ID_CACHE:
            return _TW_ENV_ID_CACHE[game_file]

        with _TW_REGISTER_LOCK:
            # Re-check inside the lock: another thread may have registered while
            # we were waiting.
            if game_file not in _TW_ENV_ID_CACHE:
                import textworld
                import textworld.gym
                from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

                request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
                tw_env_id = textworld.gym.register_games(
                    [game_file],
                    request_infos,
                    batch_size=1,
                    asynchronous=False,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    wrappers=[_make_demangler, AlfredInfos],
                )
                _TW_ENV_ID_CACHE[game_file] = tw_env_id

        return _TW_ENV_ID_CACHE[game_file]

    def _make_tw_env(self, game_file: str):
        """Instantiate and return a fresh TextWorld gym env for one game file.

        gym.make() constructs the wrapper chain and may touch the tatsu textgen
        parser during init. Serialise with _TW_PARSE_LOCK to avoid concurrent
        threads corrupting the module-level _PARSER state.
        """
        import textworld.gym
        tw_env_id = self._get_tw_env_id(game_file)
        with _TW_PARSE_LOCK:
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
        if m == 1:
            return await parent_get_response(state, prompt, *args, **kwargs)

        # Phase 8 (generation o-sharing): the m calls use the IDENTICAL `prompt`
        # (the shared observation o), so vLLM's automatic prefix caching prefills
        # o once and reuses its KV across the m decodes -- the "don't recompute o"
        # saving for generation. This is the doc's Option A (zero-code) and REQUIRES
        # the inference server to run with prefix caching enabled
        # (enable_prefix_caching=True / not --no-enable-prefix-caching).
        #
        # Option B (one n=m request, vLLM decodes m sequences off one o prefill) is
        # NOT used: the verifiers client (OpenAIChatCompletionsClient.get_response)
        # hard-asserts exactly one choice per response, so n>1 would need a
        # verifiers-core change. The auto-prefix-cache path already shares the o
        # prefill, so the n=m win over it is marginal (one request vs m, slightly
        # tighter decode batching) and deferred.
        responses = await asyncio.gather(
            *[parent_get_response(state, prompt, *args, **kwargs) for _ in range(m)]
        )

        # Uniform j*, seeded from (trajectory_id, turn) for reproducibility. A str
        # seed gives random.Random a stable (sha512-derived) state -- deterministic
        # across processes, unlike the hash-randomized built-in hash() on strings.
        # len(trajectory) is this turn's index (the step is appended afterwards).
        rng = random.Random(f"{state['trajectory_id']}:{len(state['trajectory'])}")
        j_star = rng.randrange(m)

        state["_pending_reasoning_blocks"] = list(responses)
        state["_executed_block_idx"] = j_star
        return responses[j_star]

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
            with _TW_PARSE_LOCK:
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
                # Primary: parse <action>...</action> tag
                match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
                if match:
                    return match.group(1).strip()
                # Fallback: strip <think> block and use the rest
                text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return text if text else content.strip()
        return "look"  # safe no-op if no model message found

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
        if (
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

    async def _compute_pi_hat(
        self,
        state: vf.State,
        blocks: list,
        j_star: int,
        admissible: list[str],
        a_star: str,
    ) -> float:
        """RBMC leave-one-out estimate of pi_hat(a*|o) from the m reasoning blocks.

        For each block R_j: build [o, R_j, <action>a</action>] for every admissible
        action a (re-encoding reasoning+action together so the </think>-><action>
        BPE seam is handled, and scoring from the first divergent token so every
        candidate shares the same conditioning prefix -- the shared seam token
        cancels in the softmax). Teacher-force-score the batch (one /v1/completions
        request), renormalize over A(o) to get pi(·|o, R_j), keep the conditional at
        a*. Average across blocks leaving out the executed block j*
        (rbmc_marginal_loo). Scored at T=1 / neutral so pi_hat is the raw marginal.
        """
        # rbmc ships alongside this module (pyproject include). Dual import covers
        # both the installed top-level layout and the in-repo package layout.
        try:
            from rbmc import conditional_renorm, rbmc_marginal_loo
        except ImportError:
            from environments.alfworld.rbmc import conditional_renorm, rbmc_marginal_loo

        self._ensure_tokenizer(state)

        # Union invariant: a* must be among the scored candidates (the canonical
        # helper is prime_rl.orchestrator.admissible.substitute_executed_into_admissible;
        # replicated inline here because verifiers does not depend on prime_rl).
        candidates = list(admissible)
        if a_star in candidates:
            a_star_idx = candidates.index(a_star)
        else:
            candidates.append(a_star)
            a_star_idx = len(candidates) - 1

        client = state["client"]
        model = state["model"]
        conditionals_at_astar: list[float] = []
        for block in blocks:
            o_ids = list(block.message.tokens.prompt_ids)
            content = block.message.content or ""
            reasoning_text = content.split("<action>")[0]
            base_ids = self._tokenizer.encode(reasoning_text, add_special_tokens=False)

            # Phase 8 (scoring o-sharing): every candidate prompt below is
            # o_ids + encode(reasoning + <action>a</action>), so within a block the
            # |A| prompts share the contiguous [o, R_j] prefix and across blocks
            # they share [o]. vLLM automatic prefix caching therefore prefills
            # [o, R_j] (and [o]) once and reuses it across the |A| scoring requests
            # -- the "don't recompute o" saving for scoring. REQUIRES the inference
            # server's prefix caching to be on. Risk (Phase 9 spike, DQ8): if
            # prompt_logprobs forces a full prefill recompute and defeats the cache,
            # the fallback is a custom cache-preserving /score route on the
            # inference server (not built here -- gated on the spike result).
            prompts, spans = [], []
            for a in candidates:
                full = self._tokenizer.encode(
                    reasoning_text + "<action>" + a + "</action>",
                    add_special_tokens=False,
                )
                cpl = self._common_prefix_len(base_ids, full)
                prompts.append(o_ids + full)
                spans.append((len(o_ids) + cpl, len(o_ids) + len(full)))

            logps = await self._score_continuations(client, prompts, spans, model)
            cond = conditional_renorm(logps)
            conditionals_at_astar.append(cond[a_star_idx])

        return rbmc_marginal_loo(conditionals_at_astar, exclude_idx=j_star)

    async def _score_continuations(
        self,
        client,
        prompts_token_ids: list[list[int]],
        spans: list[tuple[int, int]],
        model: str,
        temperature: float = 1.0,
    ) -> list[float]:
        """Teacher-force-score token-id prompts via native /v1/completions
        (prompt_logprobs); return the summed actual-token logprob over each span.

        Uses the token client's public token_client (base_url with trailing /v1
        stripped) so POST "/v1/completions" hits the native endpoint. The actual
        token at each scored position is known by construction, so logprobs are
        read by token-id. Returns one value per prompt, aligned to spans.
        """
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

        obs, _reward, done, infos = await asyncio.to_thread(_step)
        state["_last_admissible_commands"] = infos["admissible_commands"][0]

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


def build_dataset(
    data_path: str,
    split: str,
    curriculum: list | None = None,
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
    """
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

    if not game_files:
        raise ValueError(f"No game.tw-pddl files found under {base_path}")

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
    tokenizer_path: str | None = None,
    num_reasoning_blocks: int = 1,
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
    """
    rubric = vf.Rubric()
    rubric.add_reward_func(alfworld_reward, weight=1.0)
    rubric.add_reward_func(context_truncated, weight=0.0)
    rubric.add_reward_func(context_evictions, weight=0.0)
    rubric.add_reward_func(admissible_action_rate, weight=0.0)
    rubric.add_reward_func(format_compliance_rate, weight=0.0)

    env = ALFWorldEnvironment(
        dataset=lambda: build_dataset(data_path, split, curriculum=curriculum),
        rubric=rubric,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        log_trajectories=log_trajectories,
        tokenizer_path=tokenizer_path,
        num_reasoning_blocks=num_reasoning_blocks,
        env_id="alfworld-env",
    )
    return env
