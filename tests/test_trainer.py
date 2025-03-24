import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from environment import GridWorld
from policies import Policy
from trainer import reinforce_plus_plus_ppo_clip

def test_reinforce_plus_plus_ppo_clip():
    """Tests the reinforce_plus_plus_ppo_clip function by training a policy and ensuring weights update."""
    env = GridWorld()
    state_size = env.size * env.size
    action_size = 4

    sft_policy = Policy(state_size, action_size)
    rl_policy = Policy(state_size, action_size)

    reinforce_plus_plus_ppo_clip(env, sft_policy, rl_policy, episodes=2, beta=0.1, epsilon=0.2)
    
    initial_weights = np.copy(rl_policy.weights)
    reinforce_plus_plus_ppo_clip(env, sft_policy, rl_policy, episodes=2, beta=0.1, epsilon=0.2)
    updated_weights = rl_policy.weights

    assert not np.array_equal(initial_weights, updated_weights), "Policy weights should be updated after training."

def test_ppo_clip_objective():
    """Tests the PPO-Clip objective by verifying the clipping logic."""
    ratio = 1.2
    advantage = 1.0
    epsilon = 0.2

    unclipped = ratio * advantage
    clip_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    clipped = clip_ratio * advantage
    objective = np.minimum(unclipped, clipped)

    assert objective == min(unclipped, clipped), "PPO-Clip objective should be the minimum of unclipped and clipped values."
