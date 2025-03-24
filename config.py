class Config:
    """
    Configuration settings for the reinforcement learning environment and training parameters.
    
    Attributes:
        GRID_SIZE (int): The size of the grid world.
        STATE_SIZE (int): The total number of states in the environment.
        ACTION_SIZE (int): The number of possible actions an agent can take.
        BETA (float): KL penalty coefficient for PPO training.
        EPSILON (float): Clipping parameter for PPO.
        GAMMA (float): Discount factor for future rewards.
        LEARNING_RATE (float): Learning rate for updating policy weights.
        EPISODES (int): The total number of training episodes.
    """
    GRID_SIZE = 8
    STATE_SIZE = GRID_SIZE * GRID_SIZE
    ACTION_SIZE = 4
    BETA = 0.1
    EPSILON = 0.2
    GAMMA = 0.99
    LEARNING_RATE = 0.01
    EPISODES = 1000