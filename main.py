from environment import GridWorld
from policies import Policy
from trainer import reinforce_plus_plus_ppo_clip

def main():
    """
    Initializes the GridWorld environment and policies, then trains the model using 
    the REINFORCE++ with PPO-Clip algorithm.
    
    The trained model is saved upon completion.
    """
    env = GridWorld()
    state_size = env.size * env.size
    action_size = 4

    sft_policy = Policy(state_size, action_size)
    rl_policy = Policy(state_size, action_size)

    reinforce_plus_plus_ppo_clip(env, sft_policy, rl_policy, episodes=1000, beta=0.1, epsilon=0.2, save_model_flag=True)

if __name__ == "__main__":
    main()
