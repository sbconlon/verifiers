"""ALFWorld environment for prime-rl / verifiers.

Environment contract:
  - setup_state():   load game from state["info"]["game_file"], reset, inject initial obs
  - env_response():  parse <action> from last AssistantMessage, step game, return UserMessage
  - cleanup:         remove non-serializable tw_env from state before ZMQ serialization
  - reward:          1.0 if state["won"] else 0.0 (set by env_response on terminal step)
"""
import asyncio
import os
import re
import threading

import verifiers as vf
from datasets import Dataset

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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ALFWorldEnvironment(vf.MultiTurnEnv):

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

                request_infos = textworld.EnvInfos(won=True, admissible_commands=True)
                tw_env_id = textworld.gym.register_games(
                    [game_file],
                    request_infos,
                    batch_size=1,
                    asynchronous=False,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
                )
                _TW_ENV_ID_CACHE[game_file] = tw_env_id

        return _TW_ENV_ID_CACHE[game_file]

    def _make_tw_env(self, game_file: str):
        """Instantiate and return a fresh TextWorld gym env for one game file."""
        import textworld.gym
        tw_env_id = self._get_tw_env_id(game_file)
        return textworld.gym.make(tw_env_id)

    @staticmethod
    def _format_obs(obs: str, admissible_commands: list[str]) -> str:
        """Combine observation text with the list of available actions."""
        commands = "\n".join(admissible_commands)
        return f"{obs}\n\nAvailable actions:\n{commands}"

    async def setup_state(self, state: vf.State) -> vf.State:
        game_file = state["info"]["game_file"]
        tw_env = await asyncio.to_thread(self._make_tw_env, game_file)
        obs, infos = await asyncio.to_thread(tw_env.reset)

        state["alf_env"] = tw_env
        state["won"] = False
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

        obs, _reward, done, infos = await asyncio.to_thread(tw_env.step, [action])

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
) -> vf.Environment:
    rubric = vf.Rubric()
    rubric.add_reward_func(alfworld_reward)

    env = ALFWorldEnvironment(
        # Callable so the dataset is built lazily — after env_id is set by
        # vf.load_environment(), ensuring the task column matches resolved_name.
        dataset=lambda: build_dataset(data_path, split),
        rubric=rubric,
        max_turns=max_turns,
        env_id="alfworld",
    )
    return env
