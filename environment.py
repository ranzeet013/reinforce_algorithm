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

class GridWorld:
    """
    Represents a grid-based environment where an agent navigates to a goal.
    
    Attributes:
        size (int): The size of the grid (default is 8x8).
        goal (tuple): The coordinates of the goal position.
        agent_pos (tuple): The current position of the agent.
    """
    def __init__(self, size=8):
        """
        Initializes the grid world environment.
        
        Args:
            size (int, optional): The size of the grid. Defaults to 8.
        """
        self.size = size
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        """
        Resets the agent's position to the starting point.
        
        Returns:
            tuple: The initial position of the agent.
        """
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(self, action):
        """
        Moves the agent in the grid based on the given action.
        
        Args:
            action (int): The action to be taken (0=Up, 1=Down, 2=Left, 3=Right).
        
        Returns:
            tuple: The new position of the agent.
            int: The reward for the step taken.
            bool: Whether the episode has ended.
        """
        x, y = self.agent_pos
        if action == 0:
            x = max(x - 1, 0)
        elif action == 1:
            x = min(x + 1, self.size - 1)
        elif action == 2:
            y = max(y - 1, 0)
        elif action == 3:
            y = min(y + 1, self.size - 1)
        
        self.agent_pos = (x, y)
        done = (self.agent_pos == self.goal)
        reward = 10 if done else -1
        return self.agent_pos, reward, done
