import matplotlib.pyplot as plt
import numpy as np

# Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) is a probabilistic framework that provides
# a principled way of learning rewards while handling ambiguity in demonstrations.
# It assumes that expert trajectories are exponentially more likely under higher cumulative rewards,
# adding a layer of uncertainty (entropy) to the model.

# Core Changes for MaxEnt IRL respect to IRL
# Log-Likelihood of Trajectories:
#   MaxEnt IRL optimizes the reward function to maximize the log-likelihood of expert demonstrations under the learned policy.
# Soft Value Iteration:
#   Instead of computing deterministic values, MaxEnt IRL uses soft value iteration, where transitions are weighted probabilistically, allowing for a distribution over actions.
# Reward Update:
#   The gradients update the reward based on the difference between expert feature expectations and soft state visitation frequencies.
# Entropy Regularization:
#   Adds entropy to the optimization objective, ensuring that the resulting policy does not collapse to deterministic behavior unless necessary.


# --- Step 1: Define Environment (Unchanged) ---
class GridWorld:
    def __init__(self, size=5, obstacles=None, goal=None):
        self.size = size
        self.obstacles = obstacles if obstacles else []
        self.goal = goal if goal else (4, 4)
        self.actions = ["up", "down", "left", "right"]

    def get_next_state(self, state, action):
        x, y = state
        if action == "up":
            x = max(0, x - 1)
        if action == "down":
            x = min(self.size - 1, x + 1)
        if action == "left":
            y = max(0, y - 1)
        if action == "right":
            y = min(self.size - 1, y + 1)
        return (x, y) if (x, y) not in self.obstacles else state

    def is_goal(self, state):
        return state == self.goal


# --- Step 2: Soft Value Iteration ---
def soft_value_iteration(env, rewards, gamma=0.9, iterations=100):
    """
    Perform soft value iteration to calculate the state values and policy.
    """
    num_states = env.size * env.size
    values = np.zeros(num_states)  # Initialize values to zero
    policy = np.zeros((num_states, len(env.actions)))  # Policy for each state-action pair

    for _ in range(iterations):
        new_values = np.zeros(num_states)
        for state_idx in range(num_states):
            state = (state_idx // env.size, state_idx % env.size)
            action_values = []
            for action in env.actions:
                next_state = env.get_next_state(state, action)
                next_idx = state_to_index(next_state, env.size)
                action_values.append(rewards[state_idx] + gamma * values[next_idx])
            # Softmax over action values for stochastic policy
            max_action_value = max(action_values)  # Numerical stability
            exp_action_values = np.exp(action_values - max_action_value)
            softmax_probs = exp_action_values / np.sum(exp_action_values)
            policy[state_idx] = softmax_probs
            new_values[state_idx] = np.log(np.sum(exp_action_values)) + max_action_value
        values = new_values  # Update values

    return values, policy


# --- Step 3: MaxEnt IRL ---
def maxent_irl(env, trajectories, gamma=0.9, alpha=0.1, iterations=100):
    """
    Maximum Entropy IRL. Learns reward weights to explain expert behavior.
    """
    num_states = env.size * env.size
    features = feature_matrix(env)
    rewards = np.random.rand(num_states)  # Initialize random rewards
    feature_expectations = np.zeros(num_states)

    # Compute feature expectations from expert demonstrations
    for state, _ in trajectories:
        idx = state_to_index(state, env.size)
        feature_expectations[idx] += 1
    feature_expectations /= len(trajectories)

    for _ in range(iterations):
        # Perform soft value iteration to get the current policy
        values, policy = soft_value_iteration(env, rewards, gamma)

        # Compute state visitation frequencies using the current policy
        state_visitation = np.zeros(num_states)
        for state, _ in trajectories:
            idx = state_to_index(state, env.size)
            state_visitation[idx] += 1  # Start state
            for _ in range(100):  # Simulate trajectory rollout
                action_probs = policy[idx]
                action_idx = np.random.choice(len(env.actions), p=action_probs)
                next_state = env.get_next_state(
                    (idx // env.size, idx % env.size), env.actions[action_idx]
                )
                next_idx = state_to_index(next_state, env.size)
                state_visitation[next_idx] += 1
                if env.is_goal((next_idx // env.size, next_idx % env.size)):
                    break
        state_visitation /= np.sum(state_visitation)  # Normalize frequencies

        # Update rewards using gradient ascent
        gradients = feature_expectations - np.dot(features.T, state_visitation)
        rewards += alpha * gradients

    return rewards.reshape(env.size, env.size)


# --- Utility Functions ---
def feature_matrix(env):
    """Feature matrix where each state has a one-hot encoding."""
    num_states = env.size * env.size
    features = np.eye(num_states)
    return features


def state_to_index(state, env_size):
    """Convert (x, y) state to a single index."""
    return state[0] * env_size + state[1]


def generate_expert_trajectories(env, start, expert_policy):
    trajectories = []
    state = start
    while not env.is_goal(state):
        action = expert_policy[state]
        trajectories.append((state, action))
        state = env.get_next_state(state, action)
    return trajectories


# --- Step 4: Run MaxEnt IRL ---
# Define grid world, expert policy, and expert trajectories
env = GridWorld(size=5, obstacles=[(1, 2), (2, 2)], goal=(4, 4))
expert_policy = {
    (0, 0): "down",
    (1, 0): "down",
    (2, 0): "down",
    (3, 0): "down",
    (4, 0): "right",
    (4, 1): "right",
    (4, 2): "right",
    (4, 3): "right",
}
expert_trajectories = generate_expert_trajectories(env, start=(0, 0), expert_policy=expert_policy)

# Learn rewards using MaxEnt IRL
learned_rewards = maxent_irl(env, expert_trajectories)

# Plot the learned rewards
plt.figure(figsize=(6, 6))
plt.imshow(learned_rewards, cmap="hot", interpolation="nearest")
plt.colorbar(label="Reward")
plt.title("Learned Rewards (MaxEnt IRL)")
plt.show()
