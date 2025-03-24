import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from environment import GridWorld

def test_grid_world_reset():
    """Tests if resetting the GridWorld environment returns the initial state."""
    env = GridWorld()
    state = env.reset()
    assert state == (0, 0), "Reset should return the initial state (0, 0)."

def test_grid_world_step():
    """Tests if the GridWorld environment correctly updates state, reward, and done flag after taking steps."""
    env = GridWorld()
    env.reset()

    next_state, reward, done = env.step(3)  # Move right
    assert next_state == (0, 1), "Moving right should update the state to (0, 1)."
    assert reward == -1, "Reward should be -1 for non-goal states."
    assert not done, "Done should be False for non-goal states."

    next_state, reward, done = env.step(1)  # Move down
    assert next_state == (1, 1), "Moving down should update the state to (1, 1)."

    env.agent_pos = (7, 6)
    next_state, reward, done = env.step(3)  # Move right to reach the goal
    assert next_state == (7, 7), "Moving right should update the state to (7, 7)."
    assert reward == 10, "Reward should be 10 for reaching the goal."
    assert done, "Done should be True for reaching the goal."

def test_grid_world_boundaries():
    """Tests if the GridWorld environment correctly handles boundary conditions when moving out of bounds."""
    env = GridWorld()
    env.reset()

    next_state, reward, done = env.step(2)  # Move left
    assert next_state == (0, 0), "Moving left from (0, 0) should stay at (0, 0)."

    next_state, reward, done = env.step(0)  # Move up
    assert next_state == (0, 0), "Moving up from (0, 0) should stay at (0, 0)."
