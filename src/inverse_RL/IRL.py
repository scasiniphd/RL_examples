import matplotlib.pyplot as plt
import numpy as np


# --- Step 1: Define Environment ---
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


# --- Step 2: Generate Expert Trajectories ---
def generate_expert_trajectories(env, start, expert_policy):
    trajectories = []
    state = start
    while not env.is_goal(state):
        action = expert_policy[state]
        trajectories.append((state, action))
        state = env.get_next_state(state, action)
    return trajectories


# Expert policy (hardcoded for simplicity)
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

# Create environment and generate expert demonstrations
env = GridWorld(size=5, obstacles=[(1, 2), (2, 2)], goal=(4, 4))
expert_trajectories = generate_expert_trajectories(env, start=(0, 0), expert_policy=expert_policy)


# --- Step 3: Inverse Reinforcement Learning ---
def feature_matrix(env):
    """Feature matrix where each state has a one-hot encoding."""
    num_states = env.size * env.size
    features = np.eye(num_states)
    return features


def state_to_index(state, env_size):
    """Convert (x, y) state to a single index."""
    return state[0] * env_size + state[1]


def irl(env, trajectories, gamma=0.9, alpha=0.1, iterations=100):
    """
    Linear Programming IRL. Learns reward weights to explain expert behavior.
    """
    num_states = env.size * env.size
    features = feature_matrix(env)
    rewards = np.random.rand(num_states)  # Initialize random rewards
    feature_expectations = np.zeros(num_states)

    # Compute feature expectations from expert demonstrations
    # Feature expectations represent the average features observed in expert trajectories.
    # They are a key aspect of IRL because the reward function is inferred to make the agent's feature expectations match the expert's.
    for state, _ in trajectories:
        idx = state_to_index(state, env.size)
        feature_expectations[idx] += (
            1  # For each state in the expert trajectories, we count how often the expert visits that state.
        )
    feature_expectations /= len(
        trajectories
    )  # The counts are normalized by dividing by the number of trajectories to create an average.

    # Iteratively update rewards
    # The reward function is updated iteratively based on a policy evaluation process. For each state, the algorithm computes a value function:
    for _ in range(iterations):
        values = np.zeros(
            num_states
        )  # Values Initialization: Start with all state values as zero.
        # Policy evaluation loop
        for state_idx in range(num_states):
            state = (state_idx // env.size, state_idx % env.size)
            for action in env.actions:
                next_state = env.get_next_state(state, action)
                next_idx = state_to_index(next_state, env.size)
                values[state_idx] += gamma * values[next_idx]  # Update Values: For each state,
                # compute its value by considering all possible actions and summing the discounted
                # value of the next states. This approximates how "good" each state is under the current reward.

        # Update rewards to match feature expectations
        # The reward function is updated using gradient ascent on the difference between:
        # Expert feature expectations: What the expert demonstrates.
        # Agent feature expectations: What the current policy suggests.
        gradients = feature_expectations - np.dot(features.T, values)
        rewards += alpha * gradients

    return rewards.reshape(env.size, env.size)


learned_rewards = irl(env, expert_trajectories)

# --- Step 4: Visualize Learned Rewards ---
plt.figure(figsize=(6, 6))
plt.imshow(learned_rewards, cmap="hot", interpolation="nearest")
plt.colorbar(label="Reward")
plt.title("Learned Rewards")
plt.show()
