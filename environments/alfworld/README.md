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

### Changelog

#### v0.1.0

- Initial implementation with  (MultiTurnEnv)
- Lazy dataset build to ensure correct task column alignment
- Thread-safe gym registration via 
-  wrapper for human-readable object names
