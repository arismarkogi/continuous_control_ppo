import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import PPO_Actor, PPO_Critic
from memory import RolloutBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class PPOAgent:
    def __init__(self, state_size, action_size, seed):
        self.seed = torch.manual_seed(seed)
        
        # Hyperparameters (The standard PPO defaults that "just work")
        self.gamma = 0.99         # Discount factor
        self.lam = 0.95           # GAE smoothing parameter
        self.clip_ratio = 0.2     # PPO clipping parameter (epsilon)
        self.ppo_epochs = 10      # How many times to loop over the data
        self.batch_size = 256     # Mini-batch size for gradient descent
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.c1 = 0.5             # Value loss coefficient
        self.c2 = 0.01            # Entropy coefficient (exploration bonus)
        
        # Initialize Networks & Memory
        self.actor = PPO_Actor(state_size, action_size, seed).to(device)
        self.critic = PPO_Critic(state_size, seed).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.memory = RolloutBuffer()

    def act(self, state):
        """Passes state through networks to get actions, probabilities, and value estimates."""
        state = torch.from_numpy(state).float().to(device)
        
        # We use torch.no_grad() because we don't want to calculate gradients during gameplay
        with torch.no_grad():
            action, log_prob, _ = self.actor(state)
            value = self.critic(state)
            
        return action.cpu().numpy(), log_prob, value

    def compute_gae(self, next_value):
        """The telescoping GAE math we discussed! Calculates Advantages and Returns."""
        rewards = self.memory.rewards
        values = self.memory.values
        dones = self.memory.dones
        
        advantages = []
        gae = 0
        
        # Loop backwards through the trajectory
        for t in reversed(range(len(rewards))):
            # If the episode ended, the value of the next state is 0
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            
            # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            
            # GAE telescoping sum
            gae = delta + self.gamma * self.lam * non_terminal * gae
            advantages.insert(0, gae)
            
        # Convert lists to tensors
        advantages = torch.cat(advantages, dim=0).to(device)
        values = torch.cat(values, dim=0).to(device)
        
        # The true Return is simply the calculated Advantage + the Critic's original guess
        returns = advantages + values
        
        # Normalize advantages (standard RL trick to stabilize gradients)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def learn(self, next_state):
        """The Master PPO Update Loop."""
        # 1. Get the Critic's guess for the state AFTER the rollout ended
        next_state = torch.from_numpy(next_state).float().to(device)
        with torch.no_grad():
            next_value = self.critic(next_state)
            
        # 2. Calculate Advantages and Returns via GAE
        advantages, returns = self.compute_gae(next_value)
        
        # 3. Train for multiple epochs on this exact batch of data
        for _ in range(self.ppo_epochs):
            for states, old_actions, old_log_probs, old_values, indices in self.memory.generate_batches(self.batch_size):
                
                # Fetch the advantages and returns specifically for this mini-batch
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]
                
                # Evaluate the old actions using the NEW, currently updating network
                _, new_log_probs, entropy = self.actor(states, old_actions)
                new_values = self.critic(states).squeeze()
                
                # --- PPO ACTOR LOSS (The Clipping) ---
                # ratio = pi_theta(a|s) / pi_theta_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # --- PPO CRITIC LOSS ---
                critic_loss = nn.MSELoss()(new_values, batch_returns)
                
                # --- TOTAL LOSS ---
                loss = actor_loss + (self.c1 * critic_loss) - (self.c2 * entropy.mean())
                
                # Backpropagation
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                # Clip gradients to prevent huge spikes
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
        # 4. Throw the data in the trash so we collect fresh, on-policy data next time
        self.memory.clear()
