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
  - Token counting: one POST to vLLM /tokenize per turn on the full message list.
    Exact counts, no local tokenizer needed. If the count exceeds the budget, messages are
    popped one at a time from messages[1:] and re-counted until the budget is satisfied.
  - Truncation metrics written to state for §5 reporting:
      state["context_truncated"]          bool  — whether any eviction occurred this episode
      state["context_truncated_at_turn"]  int   — trajectory step index of first eviction
      state["context_evictions"]          int   — total messages evicted across the episode
"""
import asyncio
import logging
import os
import re
import threading

import httpx
import verifiers as vf
from datasets import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EPISODE_STEPS = 50

SYSTEM_PROMPT = (
    "You are an agent solving household tasks in a text-based environment.\n\n"
    "At each step you receive an observation describing your current location and "
    "the objects visible to you. Your task is stated in the first observation.\n\n"
    "Think through your plan step by step, then output your action using exactly "
    "this format:\n"
    "<think>\nyour reasoning\n</think>\n"
    "<action>\nyour action\n</action>\n\n"
    "Choose exactly one action from the Available actions list at the end of each observation."
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

    def __init__(self, max_context_tokens: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.max_context_tokens = max_context_tokens
        # Shared httpx client across rollouts in this env worker process.
        # Initialised lazily in _get_http_client() to avoid creating it at
        # import time (before the asyncio event loop exists).
        self._http_client: httpx.AsyncClient | None = None

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
        """Combine observation text with the list of available actions."""
        commands = "\n".join(admissible_commands)
        return f"{obs}\n\nAvailable actions:\n{commands}"

    # ------------------------------------------------------------------
    # Context window truncation
    # ------------------------------------------------------------------

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client for /tokenize calls."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    @staticmethod
    def _derive_tokenize_url(state: vf.State) -> str:
        """Derive the vLLM /tokenize URL from the inference client's base_url.

        The client's base_url is the OpenAI-compatible root, e.g.
        "http://localhost:8765/v1/". The /tokenize endpoint lives at the
        server root (not under /v1), so we strip the /v1 suffix.
        """
        base = str(state["client"].client.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base.rstrip("/") + "/tokenize"

    async def _count_tokens(self, messages: vf.Messages, state: vf.State) -> int:
        """Return the exact token count for messages via vLLM /tokenize.

        Sends the full message list with the chat template applied server-side.
        add_generation_prompt=True matches what vLLM does before inference.
        The URL is derived once per rollout and cached in state["_tokenize_url"].
        """
        if "_tokenize_url" not in state:
            state["_tokenize_url"] = self._derive_tokenize_url(state)

        # Serialise messages to plain role/content dicts.
        # The /tokenize endpoint applies the chat template server-side, so we
        # only need role and content — no extra fields.
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

        http = await self._get_http_client()
        response = await http.post(
            state["_tokenize_url"],
            json={
                "model": state["model"],
                "messages": payload_messages,
                "add_generation_prompt": True,
            },
        )
        response.raise_for_status()
        return response.json()["count"]

    async def _apply_sliding_window(
        self, messages: vf.Messages, state: vf.State
    ) -> vf.Messages:
        """Evict oldest messages until the token count fits within budget.

        messages[0] (system prompt) is always preserved.
        messages[1:] are eviction candidates, removed FIFO (oldest first).

        After each eviction the token count is re-queried exactly via /tokenize.
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
        if self.max_context_tokens <= 0:
            return messages
        try:
            return await self._apply_sliding_window(messages, state)
        except Exception as exc:
            logger.warning(
                f"Token counting failed ({exc}); skipping truncation this turn. "
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


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_path: str, split: str) -> Dataset:
    """Scan game files under data_path/split and return a HuggingFace Dataset."""
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

    rows = [
        {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
            "answer": "",
            "info": {"game_file": gf},
        }
        for gf in game_files
    ]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Entry point called by vf.load_environment("alfworld-env", **args)
# ---------------------------------------------------------------------------

def load_environment(
    data_path: str = os.path.expanduser("~/.cache/alfworld/json_2.1.1"),
    split: str = "train",
    max_turns: int = MAX_EPISODE_STEPS,
    max_context_tokens: int = -1,
) -> vf.Environment:
    rubric = vf.Rubric()
    rubric.add_reward_func(alfworld_reward)

    env = ALFWorldEnvironment(
        dataset=lambda: build_dataset(data_path, split),
        rubric=rubric,
        max_turns=max_turns,
        max_context_tokens=max_context_tokens,
        env_id="alfworld-env",
    )
    return env
