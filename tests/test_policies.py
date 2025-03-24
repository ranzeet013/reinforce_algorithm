import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from policies import Policy

def test_policy_initialization():
    """Tests if the policy initializes with the correct weight shape."""
    policy = Policy(state_size=64, action_size=4)
    assert policy.weights.shape == (64, 4), "Policy weights should have shape (state_size, action_size)."

def test_policy_get_action():
    """Tests if the policy returns a valid action."""
    policy = Policy(state_size=64, action_size=4)
    state = (0, 0)
    action = policy.get_action(state)
    assert action in [0, 1, 2, 3], "Action should be one of [0, 1, 2, 3]."

def test_policy_get_probs():
    """Tests if the policy returns valid probability distributions."""
    policy = Policy(state_size=64, action_size=4)
    state = (0, 0)
    probs = policy.get_probs(state)
    assert isinstance(probs, np.ndarray), "Probabilities should be a numpy array."
    assert len(probs) == 4, "Probabilities should have length equal to action_size."
    assert np.isclose(np.sum(probs), 1.0), "Probabilities should sum to 1."

def test_policy_safe_softmax():
    """Tests if the policy's softmax function produces valid probabilities."""
    policy = Policy(state_size=64, action_size=4)
    state = (0, 0)
    probs = policy.get_probs(state)
    assert not np.any(np.isnan(probs)), "Probabilities should not contain NaN values."
    assert not np.any(np.isinf(probs)), "Probabilities should not contain infinity values."
