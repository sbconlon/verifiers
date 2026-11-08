# alfworld-env

### Overview
- **Environment ID**: 
- **Short description**: ALFWorld household task environment (text-based interactive fiction)
- **Tags**: alfworld, multi-turn, embodied, textworld, train, eval

### Datasets
- **Primary dataset(s)**: ALFWorld  game files (3553 train, 140 valid_seen, 239 valid_unseen)
- **Source links**: [ALFWorld](https://alfworld.github.io)
- **Split sizes**: train=3553, valid_seen=140, valid_unseen=239

### Task
- **Type**: multi-turn (interactive fiction / embodied task completion)
- **Rubric overview**: Sparse binary reward — 1.0 if the agent completes the task, 0.0 otherwise

### Quickstart
Run an evaluation with default settings:



Configure model and sampling:



### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
|  | str |  | Path to ALFWorld game files |
|  | str |  | Dataset split: , ,  |
|  | int |  | Maximum steps per episode |

### Metrics
| Metric | Meaning |
| ------ | ------- |
|  | 1.0 if task completed successfully, else 0.0 |

### Training-time probe (probe capture)

Optional mode used by the prime-rl orchestrator's probe hook (plan
`training-probe`); off by default and inert on the training path.

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| probe_set | str or None | None | Probe set shipped in `probe_sets/` (e.g. `alfworld-24-v1`); the dataset becomes exactly those games, in order |
| probe_capture | bool | False | Greedy executed response + `num_reasoning_blocks` sampled blocks per turn; writes `extras["probe"]` per step (needs `num_reasoning_blocks >= 2`) |
| probe_block_temperature | float | 0.8 | Temperature for the sampled blocks (training value) |
| probe_block_min_p | float | 0.1 | `min_p` for the sampled blocks (training value) |
| oracle | str | "off" | `off` / `handcoded` (alfworld `HandCodedTWAgent` run in-env) / `planner` (engine `policy_commands`, slow) |
| store_raw_logps | bool | False | Keep the raw m x \|A\| span logprobs in the record |
| pi_hat_renorm | bool | True | True: per-block conditionals are the softmax over the admissible union; False: raw sequence probabilities, no renormalization (D-13). Applies to training and probe capture |

`extras["probe"]` fields: `schema`, `admissible_union`, `executed_idx`,
`parsed_ok`, `exec_block`, `blocks[j].{reasoning,action,cond}`,
`pi_hat_marginal`, `Z`, `raw_logps`, `window.{labels,token_count,evictions_so_far}`,
`oracle.{action,status,plan,steps_left}`. Contract: prime-rl KB
`plan/training-probe/probe-schema.md` §1.

### Changelog

#### v0.1.0

- Initial implementation with  (MultiTurnEnv)
- Lazy dataset build to ensure correct task column alignment
- Thread-safe gym registration via 
-  wrapper for human-readable object names
