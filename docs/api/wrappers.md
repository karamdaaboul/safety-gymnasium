---
title: Wrappers
---

# Wrappers

## safety_gymnasium.wrappers

All wrappers use the Safety-Gymnasium six-tuple step API,
`(observation, reward, cost, terminated, truncated, info)`.

### Training-setup wrappers

The three wrappers below reproduce the environment-side setup used by
[FCSRL](https://github.com/czp16/FCSRL), whose paper does not describe them. They are
useful on their own, and necessary if you want to compare against those results.

```{eval-rst}
.. autoclass:: safety_gymnasium.wrappers.SafeActionRepeat
.. autoclass:: safety_gymnasium.wrappers.SafeGoalMetTerminal
.. autoclass:: safety_gymnasium.wrappers.SafePixelObservation
```

The intended stack, innermost first:

```python
import safety_gymnasium
from safety_gymnasium.wrappers import (
    SafeActionRepeat,
    SafeGoalMetTerminal,
    SafeNormalizeObservation,
    SafePixelObservation,
)

env = safety_gymnasium.make('SafetyPointGoal1-v0')  # applies SafeTimeLimit(1000)
env = SafeGoalMetTerminal(env)
env = SafeActionRepeat(env, 4)
env = SafeNormalizeObservation(env)                 # or SafePixelObservation(env, size=(64, 64))
```

Two ordering points matter:

- The time limit must stay **inside** the action repeat, which
  {func}`safety_gymnasium.make` gives you for free. It then counts simulator steps, so a
  1000 step environment becomes a 250 agent step episode covering the same 1000 simulator
  steps, and episode return and cost stay comparable to the unrepeated benchmark.
- {class}`~safety_gymnasium.wrappers.SafePixelObservation` goes **outside** the action
  repeat, so the camera is rendered once per agent step rather than once per simulator
  step.

`SafeGoalMetTerminal` may be placed on either side of the repeat: `SafeActionRepeat`
preserves and combines `goal_met`, so both orders give the same result.

#### Running the stack in parallel

Pass the stack to {func}`safety_gymnasium.vector.make` as a per-environment `wrappers`
callable:

```python
venv = safety_gymnasium.vector.make(
    'SafetyPointGoal1-v0',
    num_envs=8,
    asynchronous=True,
    wrappers=lambda env: SafeActionRepeat(SafeGoalMetTerminal(env), 4),
)
obs, rewards, costs, terminateds, truncateds, infos = venv.step(actions)
```

Sub-environments are auto-reset within the step that ends an episode, so on that step
`infos` holds `final_observation` and `final_info` and every *other* key comes from
`reset`. That means `sim_steps`, `cost_sum` and `goal_met` are absent on episode
boundaries and live under `infos['final_info']` instead. Read them with `infos.get(...)`
or step accounting will silently lose one agent step per episode.

#### Agent steps are not simulator steps

With an action repeat of `n`, a run reported as N steps consumes `n`N simulator steps.
`SafeActionRepeat` puts the number of simulator steps each agent step actually consumed in
`info['sim_steps']` so a training loop can log both. FCSRL reports 2M steps, which is 8M
simulator steps.

#### Differences from the FCSRL reference implementation

- **`info` is preserved.** FCSRL's action repeat returns a fresh dict holding only `cost`
  and `terminate`, discarding `goal_met`, `cost_sum` and the per-constraint `cost_*` keys.
  `SafeActionRepeat` sums the `cost_*` entries, combines `goal_met` with logical `or`, and
  keeps the most recent value of everything else. Ignore the extra keys and the training
  signal is identical.
- **The pseudo-terminal includes real terminations.** FCSRL sets its flag from `goal_met`
  alone, so a genuine termination is stored as non-terminal and gets bootstrapped through.
  {class}`~safety_gymnasium.wrappers.SafeGoalMetTerminal` uses `terminated or goal_met`.
- **Pixels do not touch global state.** FCSRL calls the private `task._obs_vision()` and
  mutates the `VisionEnvConf.vision_size` class attribute.
  {class}`~safety_gymnasium.wrappers.SafePixelObservation` calls the public
  {meth}`safety_gymnasium.bases.underlying.Underlying.render` with an explicit size.

### Cost-limit curriculum

A safe-RL curriculum starts with a loose constraint budget so the agent can learn the
task at all, then tightens it as competence grows. The wrapper below tracks mean goals
per episode over a window of completed episodes and steps a cost limit down each time
the mean reaches a threshold, clamped at a minimum. It is *expose-only*: nothing in the
environment's returns changes. The training loop reads `env.cost_limit` (or
`info['cost_limit']`, present on every reset and step) and feeds it to its constraint,
for example a Lagrangian budget.

```{eval-rst}
.. autoclass:: safety_gymnasium.wrappers.SafeCostLimitCurriculum
```

```python
env = safety_gymnasium.make('SafetyPointGoal1-v0')
env = SafeCostLimitCurriculum(
    env,
    initial_cost_limit=100.0,
    min_cost_limit=25.0,
    decrement=15.0,
    success_threshold=1.5,
    window=20,
)
env = SafeGoalMetTerminal(env)
env = SafeActionRepeat(env, 4)
```

Persist the curriculum with the policy, or a restored checkpoint will resume at the
initial limit:

```python
checkpoint['curriculum'] = env.get_state()
...
env.set_state(checkpoint['curriculum'])
```

#### Placement relative to action repeat and autoreset

Place the wrapper **inside** {class}`~safety_gymnasium.wrappers.SafeActionRepeat`, as in
the snippet above, so it observes every simulator step's `goal_met`. Outside the repeat,
`goal_met` is OR-collapsed per agent step, so two goals reached within one repeat would
count as one. Its `info['cost_limit']` still reaches the outer stack, because the limit
only changes on the episode-ending simulator step, which is always the repeat's last.

The wrapper also works outside {class}`~safety_gymnasium.wrappers.SafeAutoResetWrapper`:
on the auto-resetting boundary step it reads the terminal step's `goal_met` from
`info['final_info']`.

### Normalization

```{eval-rst}
.. autoclass:: safety_gymnasium.wrappers.SafeNormalizeObservation
.. autoclass:: safety_gymnasium.wrappers.SafeNormalizeReward
.. autoclass:: safety_gymnasium.wrappers.SafeNormalizeCost
```

Statistics must not move during evaluation. Share one normalizer between the training and
evaluation environments and freeze the latter:

```python
train_env = SafeNormalizeObservation(safety_gymnasium.make(env_id))
eval_env = SafeNormalizeObservation(safety_gymnasium.make(env_id))
eval_env.normalizer = train_env.normalizer
eval_env.freeze()
```

`freeze()` is per wrapper, not per normalizer, so freezing the evaluation environment
leaves the training environment updating the shared statistics.

Persist the statistics with the policy, or a restored checkpoint will see a different input
distribution than it was trained on:

```python
checkpoint['obs_stats'] = train_env.get_stats()
...
train_env.set_stats(checkpoint['obs_stats'])
```

If your training loop owns normalization itself, for example because it keeps raw
observations in a replay buffer and re-normalizes at sample time, use the underlying
object directly instead of the wrapper:

```{eval-rst}
.. autoclass:: safety_gymnasium.utils.normalizer.MeanStdNormalizer
.. autoclass:: safety_gymnasium.utils.normalizer.RunningMeanStd
```

### API conversion

```{eval-rst}
.. autoclass:: safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium
.. autoclass:: safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium
.. autofunction:: safety_gymnasium.wrappers.with_gymnasium_wrappers
```

### Applied by `safety_gymnasium.make`

```{eval-rst}
.. autoclass:: safety_gymnasium.wrappers.SafeTimeLimit
.. autoclass:: safety_gymnasium.wrappers.SafeAutoResetWrapper
.. autoclass:: safety_gymnasium.wrappers.SafePassiveEnvChecker
```

### Other

```{eval-rst}
.. autoclass:: safety_gymnasium.wrappers.SafeRescaleAction
.. autoclass:: safety_gymnasium.wrappers.SafeUnsqueeze
```
