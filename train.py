import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import PPOAgent

env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
state_size = env_info.vector_observations.shape[1]
action_size = brain.vector_action_space_size

# Initialize our custom PPO Agent
agent = PPOAgent(state_size=state_size, action_size=action_size, seed=42)

def train_ppo(n_episodes=300, max_t=1000):
    scores_window = deque(maxlen=100) # Tracks the last 100 episodes
    all_scores = []                   # Tracks every episode for the final plot
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        episode_scores = np.zeros(num_agents)
        
        # 1. Collect a Trajectory (Rollout)
        for t in range(max_t):
            # Ask the agent what to do
            actions, log_probs, values = agent.act(states)
            
            # Send actions to the Unity environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            # Format the data for PyTorch and store it in our buffer
            agent.memory.store(
                torch.tensor(states).float().to(device),
                torch.tensor(actions).float().to(device),
                log_probs,
                torch.tensor(rewards).float().to(device).unsqueeze(1),
                values,
                torch.tensor(dones).float().to(device).unsqueeze(1)
            )
            
            states = next_states
            episode_scores += rewards
            
            if np.any(dones):
                break
                
        # 2. Update the Agent
        # Pass the final next_states so GAE can calculate the very last bootstrap value
        agent.learn(next_states)
        
        # 3. Track Scoring
        # Udacity requires you to take the mean of all 20 agents for the episode score
        mean_score = np.mean(episode_scores)
        scores_window.append(mean_score)
        all_scores.append(mean_score)
        
        print(f'\rEpisode {i_episode}\tAverage Score (Last 100): {np.mean(scores_window):.2f}\tCurrent Score: {mean_score:.2f}', end="")
        
        if i_episode % 10 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score (Last 100): {np.mean(scores_window):.2f}')
            
        if np.mean(scores_window) >= 30.0:
            print(f'\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
            break
            
    return all_scores

# Run the training loop!
scores = train_ppo()

# --- Plotting for the Final Report ---
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, color='teal', label='Score per Episode')

# Add a moving average line (looks great on reports)
import pandas as pd
moving_avg = pd.Series(scores).rolling(window=10).mean()
plt.plot(np.arange(1, len(scores)+1), moving_avg, color='orange', linewidth=2, label='10-Episode Moving Average')

# Add the target line
plt.axhline(y=30, color='r', linestyle='--', label='Target Score (+30)')

plt.title('PPO Training Progress: 20-Agent Reacher')
plt.ylabel('Average Score (Over 20 Agents)')
plt.xlabel('Episode #')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Save the plot for your submission
fig.savefig("PPO_scores_plot.png", dpi=300)
