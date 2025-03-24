import numpy as np
import logging
import os
from datetime import datetime

def configure_logging():
    """
    Configures logging settings, ensuring logs are saved in the 'logs' directory.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logging.basicConfig(
        filename=f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

def save_model(policy, filename):
    """
    Saves the model weights to the 'models' directory.
    
    Args:
        policy: The policy object containing weights to be saved.
        filename (str): The name of the file to save the model.
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    np.save(f"models/{filename}", policy.weights)
    logging.info(f"Model saved to models/{filename}.npy")

def load_model(policy, filename):
    """
    Loads model weights from a specified file.
    
    Args:
        policy: The policy object to load weights into.
        filename (str): The name of the file containing saved weights.
    """
    policy.weights = np.load(f"models/{filename}.npy")
    logging.info(f"Model loaded from models/{filename}.npy")

def reinforce_plus_plus_ppo_clip(env, sft_policy, rl_policy, episodes=1000, beta=0.1, gamma=0.99, epsilon=0.2, save_model_flag=True):
    """
    Trains the RL policy using REINFORCE++ with PPO-Clip algorithm.
    
    Args:
        env: The environment instance.
        sft_policy: The supervised fine-tuning policy.
        rl_policy: The reinforcement learning policy.
        episodes (int): Number of training episodes.
        beta (float): KL divergence penalty coefficient.
        gamma (float): Discount factor for rewards.
        epsilon (float): Clipping parameter for PPO.
        save_model_flag (bool): Whether to save the trained model.
    """
    grid_size = env.size
    for episode in range(episodes):
        state = env.reset()
        done = False
        rewards = []
        actions = []
        states = []
        old_probs = []

        while not done:
            action = rl_policy.get_action(state)
            next_state, reward, done = env.step(action)
            
            sft_probs = sft_policy.get_probs(state)
            rl_probs = rl_policy.get_probs(state)
            kl_div = np.sum(rl_probs * np.log((rl_probs + 1e-10) / (sft_probs + 1e-10)))
            reward = reward - beta * kl_div
            
            rewards.append(reward)
            actions.append(action)
            states.append(state)
            old_probs.append(rl_probs[action])
            state = next_state

        total_reward = sum(rewards)
        logging.info(f"Episode {episode + 1}, Total Reward: {total_reward}")

        G = 0
        returns = []
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        advantages = returns - np.mean(returns)

        for t in range(len(rewards)):
            state = states[t]
            action = actions[t]
            state_index = state[0] * grid_size + state[1]
            new_probs = rl_policy.get_probs(state)
            new_prob = new_probs[action]
            old_prob = old_probs[t]
            ratio = new_prob / (old_prob + 1e-10)
            clip_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
            objective = np.minimum(ratio * advantages[t], clip_ratio * advantages[t])
            grad = -objective * np.log(new_prob + 1e-10)
            rl_policy.weights[state_index] += 0.01 * grad

    if save_model_flag:
        save_model(rl_policy, "rl_policy")
