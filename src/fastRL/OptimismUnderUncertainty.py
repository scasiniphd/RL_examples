import gym
import numpy as np
import pybullet_envs  # Import PyBullet environments
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.optimize import minimize

# Load the CartPoleContinuousBulletEnv environment
pybullet_envs.register()

# Define the environment
env = gym.make("CartPoleContinuousBulletEnv-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # Use scalar for action bounds


def build_dynamics_model(state_dim, action_dim):
    """Build a neural network for the dynamics model."""
    model = Sequential(
        [
            Dense(128, activation="relu", input_dim=state_dim + action_dim),
            Dense(128, activation="relu"),
            Dense(state_dim, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def optimistic_action_selection(state, dynamics_model, action_bound, num_samples=100):
    """Select actions using Optimism Under Uncertainty."""
    actions = np.random.uniform(-action_bound, action_bound, (num_samples, action_dim))
    state_action_pairs = np.hstack([np.tile(state, (num_samples, 1)), actions])

    # Predict next states
    predicted_states = dynamics_model.predict(state_action_pairs)

    # Measure uncertainty (variance as a proxy for uncertainty)
    uncertainties = np.var(predicted_states, axis=1)

    # Choose the action with the highest uncertainty
    best_action = actions[np.argmax(uncertainties)]
    return best_action


def optimize_action_sequence(state, dynamics_model, target_position, horizon=10):
    """Optimize a sequence of actions to minimize cost."""

    def trajectory_cost(actions):
        total_cost = 0
        simulated_state = state.copy()
        actions = actions.reshape(horizon, action_dim)

        for action in actions:
            state_action = np.hstack([simulated_state, action])
            simulated_state = dynamics_model.predict(state_action[np.newaxis])[0]
            total_cost += np.linalg.norm(simulated_state[:2] - target_position)  # Example cost

        return total_cost

    # Initial random actions
    init_actions = np.random.uniform(-action_bound, action_bound, (horizon, action_dim)).flatten()

    result = minimize(
        trajectory_cost, init_actions, bounds=[(-action_bound, action_bound)] * len(init_actions)
    )
    return result.x[:action_dim]  # Return the first action


# Initialize the dynamics model
dynamics_model = build_dynamics_model(state_dim, action_dim)

# Hyperparameters
episodes = 50
steps_per_episode = 100
replay_buffer = []
target_position = np.array([0.0, 0.0])  # Example target in 2D

# Training loop
for episode in range(episodes):
    state = env.reset()
    for step in range(steps_per_episode):
        # Select action using exploration strategy
        action = optimistic_action_selection(state, dynamics_model, action_bound)

        # Perform the action in the environment
        next_state, reward, done, _ = env.step(action)

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, next_state))

        # Train the dynamics model
        if len(replay_buffer) > 100:
            batch = np.random.choice(replay_buffer, 64)
            states, actions, rewards, next_states = zip(*batch)
            dynamics_model.train_on_batch(np.hstack([states, actions]), next_states)

        state = next_state
        if done:
            break

    print(f"Episode {episode + 1}: Completed")


# Visualization
def visualize_policy(env, dynamics_model, target_position, steps=100):
    """Visualize the robotic arm following the learned policy."""
    state = env.reset()
    for step in range(steps):
        action = optimize_action_sequence(state, dynamics_model, target_position)
        state, _, done, _ = env.step(action)
        env.render()
        if done:
            break


# Run visualization
visualize_policy(env, dynamics_model, target_position)
env.close()
