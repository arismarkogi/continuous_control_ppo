import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        """Saves a single step of experience for all 20 agents."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """Empties the buffer after the network has updated."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def generate_batches(self, batch_size):
        """Flattens the 20-agent trajectories and chunks them into mini-batches."""
        # Stack lists of arrays into single massive tensors
        states = torch.cat(self.states, dim=0)
        actions = torch.cat(self.actions, dim=0)
        log_probs = torch.cat(self.log_probs, dim=0)
        values = torch.cat(self.values, dim=0)
        
        # Calculate the total number of experiences (e.g., 1000 steps * 20 agents = 20,000)
        total_samples = states.shape[0]
        
        # Generate an array of shuffled indices
        indices = np.arange(total_samples, dtype=np.int64)
        np.random.shuffle(indices)
        
        # Yield mini-batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                values[batch_indices],
                batch_indices # We return indices to map advantages/returns later
            )
