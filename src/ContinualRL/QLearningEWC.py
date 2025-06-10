import matplotlib.pyplot as plt
import numpy as np


# Grid World Environment
class GridWorld:
    def __init__(self, size, goal, rewards: list[float] = [1, -1]):
        self.size = size
        self.goal = goal
        self.rewards = rewards
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)

    def step(self, action):
        # Actions: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        x, y = self.agent_pos
        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.size - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.size - 1, y + 1)

        self.state = (x, y)
        reward = self.rewards[1]
        done = self.state == self.goal
        if done:
            reward = self.rewards[0]
        return self.state, reward, done, {}


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_space, action_space):
        self.q_table = {state: [0] * action_space for state in state_space}
        self.action_space = action_space
        self.epsilon = 0.5  # Exploration-exploitation tradeoff

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_space)  # random policy
        else:
            return np.argmax(self.q_table[state])


# Elastic Weight Consolidation (EWC)
class EWC:
    def __init__(self, importance_scale=0.01):
        self.fisher_information = {}
        self.optimal_params = {}
        self.importance_scale = importance_scale

    def compute_fisher_information(self, q_table):
        # Approximating the Fisher Information using Q-values
        self.fisher_information = {state: np.square(actions) for state, actions in q_table.items()}
        self.optimal_params = {state: actions.copy() for state, actions in q_table.items()}

    def apply_ewc(self, q_table):
        penalty = 0.0
        for state, fisher_values in self.fisher_information.items():
            if state in q_table:
                penalty += np.sum(
                    fisher_values
                    * np.square(np.array(q_table[state]) - np.array(self.optimal_params[state]))
                )
        return penalty * self.importance_scale


# Helper: Generate a state space
def generate_state_space(size):
    return [(x, y) for x in range(size) for y in range(size)]


# Training Process
def train(
    env,
    agent,
    ewc=None,
    num_episodes=1000,
    learning_rate=0.99,
    gamma=0.99,
    regularization_strength=0.5,
):
    rewards = []
    state = env.reset()
    done = False
    total_reward = 0
    for episode in range(num_episodes):
        if not done:
            # Choose an action
            action = agent.choose_action(state)

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Q-Learning update
            old_q_value = agent.q_table[state][action]
            max_next_q = max(agent.q_table[next_state])
            agent.q_table[state][action] = old_q_value + learning_rate * (
                reward + gamma * max_next_q - old_q_value
            )

            # Move to the next state
            state = next_state

        # Apply EWC regularization penalty if available
        if ewc is not None:
            ewc_penalty = ewc.apply_ewc(agent.q_table)
            print(ewc_penalty)
            total_reward -= regularization_strength * ewc_penalty

        # Track rewards
        rewards.append(total_reward)

        # Decay epsilon
        agent.epsilon = max(0.1, agent.epsilon * 0.995)

        # Debug output every 100 episodes
        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
            )

    return rewards


# Test the agent
def test_agent(env, agent, episodes=500):
    state = env.reset()
    done = False
    total_reward = 0
    for episode in range(episodes):
        if not done:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
    return total_reward


def plot_policy_grid(agent, env, env_name: str):
    # Define action mapping to arrows
    action_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    # Grid size
    grid_size = env.size  # Assuming square grid environment

    # Create a grid for the policy
    policy_grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

    # Fill the grid with the best actions
    for state, actions in agent.q_table.items():
        x, y = state
        best_action = np.argmax(actions)
        policy_grid[x][y] = action_map[best_action]

    # Plot the grid
    plt.figure(figsize=(6, 6))
    plt.title("Policy Grid (Best Actions by Max Q-Value)")
    plt.imshow(np.zeros((grid_size, grid_size)), cmap="Greys", origin="upper")  # Background grid

    # Overlay arrows for best actions
    for x in range(grid_size):
        for y in range(grid_size):
            plt.text(y, x, policy_grid[x][y], ha="center", va="center", color="black", fontsize=16)

    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(color="black", linestyle="-", linewidth=0.5)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig("./figures/policy_grid_" + env_name + ".png", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # np.random.seed(42)
    grid_size = 5

    # Task 1: Goal at bottom-right
    env1 = GridWorld(size=grid_size, goal=(grid_size - 1, grid_size - 1))
    state_space = generate_state_space(grid_size)
    agent = QLearningAgent(state_space=state_space, action_space=4)
    ewc = EWC(importance_scale=1000)

    print("Training on Task 1...")
    train(env1, agent, num_episodes=500)

    # Compute Fisher Information for Task 1
    ewc.compute_fisher_information(agent.q_table)

    # Task 2: New goal at top-right
    env2 = GridWorld(size=grid_size, goal=(0, grid_size - 1))
    print("Training on Task 2 with EWC...")
    train(env2, agent, ewc=ewc, num_episodes=500)
