import numpy as np
from utils import safe_softmax

class Policy:
    """
    Represents a policy for selecting actions based on learned weights.
    
    Attributes:
        state_size (int): The total number of states.
        action_size (int): The number of possible actions.
        weights (numpy.ndarray): A matrix of weights for action selection.
    """
    def __init__(self, state_size, action_size):
        """
        Initializes the policy with random small weights.
        
        Args:
            state_size (int): The total number of states.
            action_size (int): The number of possible actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.weights = np.random.rand(state_size, action_size) * 0.01

    def get_action(self, state):
        """
        Selects an action using a softmax probability distribution.
        
        Args:
            state (tuple): The (x, y) coordinates of the current state.
        
        Returns:
            int: The chosen action.
        """
        grid_size = int(np.sqrt(self.state_size))
        state_index = state[0] * grid_size + state[1]
        probs = safe_softmax(self.weights[state_index])
        return np.random.choice(self.action_size, p=probs)

    def get_probs(self, state):
        """
        Computes the action probabilities using the softmax function.
        
        Args:
            state (tuple): The (x, y) coordinates of the current state.
        
        Returns:
            numpy.ndarray: Probability distribution over actions.
        """
        grid_size = int(np.sqrt(self.state_size))
        state_index = state[0] * grid_size + state[1]
        return safe_softmax(self.weights[state_index])
