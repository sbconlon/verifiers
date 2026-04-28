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
import re
import threading
from pathlib import Path

import verifiers as vf
from datasets import Dataset

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
    "\n\nNow it's your turn to take an action. "
    "You MUST enclose your reasoning within <think> </think> tags, "
    "then choose one action from the list above and enclose it within "
    "<action> </action> tags."
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
# Environment
# ---------------------------------------------------------------------------

class ALFWorldEnvironment(vf.MultiTurnEnv):

    def __init__(self, max_context_tokens: int = -1, log_trajectories: str = "none", **kwargs):
        super().__init__(**kwargs)
        self.max_context_tokens = max_context_tokens
        self.log_trajectories = log_trajectories  # "none" | "wins" | "all"
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

    async def _count_tokens(self, messages: vf.Messages, state: vf.State) -> int:
        """Return the exact token count for messages using a local HF tokenizer.

        Lazily loads the HuggingFace tokenizer for state["model"] on first call
        (~200 MB, one-time cost per env worker process; reused across all rollouts
        on this worker). Applies the model's chat template with
        add_generation_prompt=True to match what vLLM does server-side before
        inference — i.e. the count returned here equals what vLLM will see when
        it receives the same message list.

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
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            model_name = state["model"]
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(
                f"Loaded local tokenizer for {model_name} "
                f"(env worker pid={os.getpid()})"
            )

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

    async def setup_state(self, state: vf.State) -> vf.State:
        game_file = state["info"]["game_file"]
        logger.debug(f"setup_state: loading {game_file}")
        tw_env = await asyncio.to_thread(self._make_tw_env, game_file)

        def _reset():
            with _TW_PARSE_LOCK:
                return tw_env.reset()

        try:
            obs, infos = await asyncio.to_thread(_reset)
        except Exception as exc:
            # TextWorld 1.7.0 textgen parser fails on some game file templates.
            # Re-raise as vf.Error so the framework catches it, records the
            # episode as failed (reward 0.0), and keeps the worker alive.
            logger.warning(f"setup_state: reset() failed for {game_file!r}: {exc}")
            try:
                tw_env.close()
            except Exception:
                pass
            raise vf.Error(f"setup_state failed for {game_file!r}: {exc}") from exc

        state["alf_env"] = tw_env
        state["won"] = False
        # Truncation metrics — initialised here so they are always present in
        # state even for episodes where truncation never fires.
        state["context_truncated"] = False
        state["context_truncated_at_turn"] = None
        state["context_evictions"] = 0
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

        def _step():
            with _TW_PARSE_LOCK:
                return tw_env.step([action])

        obs, _reward, done, infos = await asyncio.to_thread(_step)

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
) -> vf.Environment:
    rubric = vf.Rubric()
    rubric.add_reward_func(alfworld_reward, weight=1.0)
    rubric.add_reward_func(context_truncated, weight=0.0)
    rubric.add_reward_func(context_evictions, weight=0.0)

    env = ALFWorldEnvironment(
        dataset=lambda: build_dataset(data_path, split, curriculum=curriculum),
        rubric=rubric,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        log_trajectories=log_trajectories,
        env_id="alfworld-env",
    )
    return env
