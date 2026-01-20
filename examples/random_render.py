#!/usr/bin/env python3
"""Run a random policy with human rendering."""

import argparse

import safety_gymnasium


def run_random(env_name: str, seed: int | None) -> None:
    """Run random actions forever, rendering each step."""
    env = safety_gymnasium.make(env_name, render_mode="human")
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    ep_ret = 0.0
    ep_cost = 0.0

    while True:
        if terminated or truncated:
            print(f"Episode Return: {ep_ret}\tEpisode Cost: {ep_cost}")
            ep_ret = 0.0
            ep_cost = 0.0
            obs, _ = env.reset()

        assert env.observation_space.contains(obs)
        action = env.action_space.sample()
        assert env.action_space.contains(action)

        obs, reward, cost, terminated, truncated, _ = env.step(action)
        ep_ret += reward
        ep_cost += cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SafetyPointFormulaOne2Debug-v0")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_random(args.env, args.seed)


if __name__ == "__main__":
    main()

